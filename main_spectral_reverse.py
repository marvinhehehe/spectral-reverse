import os
import sys
import json
import argparse
import random
import math
import hashlib
import importlib
from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
from torch.func import functional_call as _functional_call
from typing import Dict, List, Optional, Any

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    AlphaEditHyperParams,
    WISEHyperParams,
    BaseEditor,
)
from easyeditor.models.prune import PRUNE
from easyeditor.models.simie import SimIE
from easyeditor.editors.utils import summary_metrics
from easyeditor.util import nethook
from easyeditor.util.generate import generate_fast
from easyeditor.models.alphaedit import apply_AlphaEdit_to_model
alphaedit_main = importlib.import_module("easyeditor.models.alphaedit.AlphaEdit_main")


true_dir = "."
CONTEXT_TEMPLATES_CACHE = None

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _take(xs, idxs):
    if xs is None:
        return None
    return [xs[i] for i in idxs]


def _slice_locality_inputs(locality_inputs, idxs):
    if locality_inputs is None:
        return None
    out = {}
    for k, v in locality_inputs.items():
        out[k] = {
            'prompt': _take(v['prompt'], idxs),
            'ground_truth': _take(v['ground_truth'], idxs)
        }
    return out


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]]
        return CONTEXT_TEMPLATES_CACHE
        gen = generate_fast(
            model,
            tok,
            ["The", "Therefore", "Because", "I", "You"],
            n_gen_per_prompt=1,
            max_out_len=10,
        )
        ctx = [g.replace("{", " ").replace("}", " ") + ". {}" for g in gen]
        CONTEXT_TEMPLATES_CACHE = [["{}"], ctx]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE

def _build_context_expanded_prompts(model, tok, prompts: List[str], subjects: List[str]):
    """
    Build the exact context-expanded prompt instances used during spectral reverse.
    """
    assert len(prompts) == len(subjects), "prompts and subjects length mismatch"

    prompts_norm = deepcopy(prompts)
    for i, (subject, prompt) in enumerate(zip(subjects, prompts_norm)):
        if '{}' not in prompt:
            assert subject in prompt, f"Subject:{subject} not in prompt:{prompt}"
            prompts_norm[i] = prompt.replace(subject, '{}')

    context_templates = get_context_templates(model, tok)
    flat_contexts = [c for ctx_group in context_templates for c in ctx_group]

    prompt_templates = [
        ctx.format(p) for p in prompts_norm for ctx in flat_contexts
    ]
    template_subjects = [s for s in subjects for _ in flat_contexts]
    actual_prompts = [tmpl.format(s) for tmpl, s in zip(prompt_templates, template_subjects)]

    return flat_contexts, prompt_templates, template_subjects, actual_prompts


def _weight_to_canon(W_param: torch.Tensor, d_k: int):
    """
    Return (W_canon, transposed_flag) where W_canon is [d_v, d_k].
    Works for Linear ([d_v, d_k]) and GPT2 Conv1D-style ([d_k, d_v]).
    """
    if W_param.ndim != 2:
        raise RuntimeError(f"Expected 2D weight, got {W_param.ndim}D")

    # already [d_v, d_k]
    if W_param.shape[1] == d_k:
        return W_param, False

    # Conv1D-style: [d_k, d_v] -> transpose to [d_v, d_k]
    if W_param.shape[0] == d_k:
        return W_param.t().contiguous(), True

    raise RuntimeError(f"Cannot infer canon shape from {tuple(W_param.shape)} with d_k={d_k}")


def _canon_to_param(M_canon: torch.Tensor, transposed: bool):
    """Inverse of _weight_to_canon for matrices shaped [d_v, d_k]."""
    return M_canon.t().contiguous() if transposed else M_canon


@torch.no_grad()
def snapshot_rewrite_weights(model, hparams) -> Dict[str, torch.Tensor]:
    """Snapshot rewrite weights (+bias if present) for hparams.layers."""
    out: Dict[str, torch.Tensor] = {}
    for layer in getattr(hparams, 'layers', []):
        mod_name = hparams.rewrite_module_tmp.format(layer)
        # weight is always present
        w_name = mod_name + '.weight'
        out[w_name] = nethook.get_parameter(model, w_name).detach().cpu().clone()
        # bias is optional
        try:
            b_name = mod_name + '.bias'
            out[b_name] = nethook.get_parameter(model, b_name).detach().cpu().clone()
        except LookupError:
            pass
    return out


@torch.no_grad()
def load_rewrite_weights(model, weights: Dict[str, torch.Tensor], device=None):
    """Load a snapshot created by snapshot_rewrite_weights into model (in-place)."""
    device = device if device is not None else next(model.parameters()).device
    for name, t in weights.items():
        try:
            p = nethook.get_parameter(model, name)
        except LookupError:
            continue
        p.data.copy_(t.to(device=device, dtype=p.dtype))


@torch.no_grad()
def compute_prompt_module_inout(
    model,
    prompt_inputs: List[Dict[str, torch.Tensor]],
    layer: int,
    module_template: str,
    batch_size: int = 64,
):
    """Return per-prompt module inputs/outputs for every prompt token."""
    module_name = module_template.format(layer)
    device = next(model.parameters()).device
    inp_all, out_all = [None] * len(prompt_inputs), [None] * len(prompt_inputs)

    def _first_tensor(x):
        while isinstance(x, (tuple, list)):
            x = x[0]
        return x

    by_len = {}
    for i, enc in enumerate(prompt_inputs):
        by_len.setdefault(int(enc["input_ids"].size(1)), []).append(i)

    for seq_len, idxs in by_len.items():
        for start in range(0, len(idxs), max(1, int(batch_size))):
            batch_idxs = idxs[start:start + max(1, int(batch_size))]
            enc_d = {
                k: torch.cat([prompt_inputs[i][k] for i in batch_idxs], dim=0).to(device)
                for k in prompt_inputs[batch_idxs[0]]
            }
            bsz = len(batch_idxs)
            with nethook.Trace(
                model,
                module_name,
                retain_input=True,
                retain_output=True,
                clone=True,
                detach=True,
            ) as tr:
                model(**enc_d)

            l_inp = _first_tensor(tr.input)
            l_out = _first_tensor(tr.output)
            if l_inp.dim() == 2 and l_inp.size(0) == bsz * seq_len:
                l_inp = l_inp.view(bsz, seq_len, -1)
            if l_out.dim() == 2 and l_out.size(0) == bsz * seq_len:
                l_out = l_out.view(bsz, seq_len, -1)
            if l_inp.dim() != 3:
                raise RuntimeError(f"Unexpected module input dim={l_inp.dim()} for {module_name}")
            if l_out.dim() != 3:
                raise RuntimeError(f"Unexpected module output dim={l_out.dim()} for {module_name}")

            for row, prompt_idx in enumerate(batch_idxs):
                inp_all[prompt_idx] = l_inp[row, :seq_len, :].float().cpu()
                out_all[prompt_idx] = l_out[row, :seq_len, :].float().cpu()

    return inp_all, out_all
    

def _find_last_subject_token_index(tok, prompt: str, subject: str) -> Optional[int]:
    """
    Find the index (0-based) of the last token of `subject` in the tokenized `prompt`.
    Handles GPT-style leading-space tokenization by trying both `subject` and `" "+subject`.
    Returns None if not found.
    """
    try:
        ids = tok(prompt, add_special_tokens=False).input_ids
    except TypeError:
        ids = tok(prompt).input_ids
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    subj_cands = [subject, " " + subject]
    matches = []
    for s in subj_cands:
        try:
            sid = tok(s, add_special_tokens=False).input_ids
        except TypeError:
            sid = tok(s).input_ids
        if isinstance(sid, torch.Tensor):
            sid = sid.tolist()
        if len(sid) == 0:
            continue
        # find all occurrences
        for i in range(0, len(ids) - len(sid) + 1):
            if ids[i:i+len(sid)] == sid:
                matches.append(i + len(sid) - 1)
    if not matches:
        return None
    return max(matches)


def spectral_gate_reverse_optimize(
    *,
    edited_model,
    tok,
    hparams,
    pre_rewrite_weights=None,
    prompts: List[str],              
    subjects: List[str],
    layers: Optional[List[int]] = None,
    rank: int = 64,
    key_batch_size: int = 64,
    fact_batch_size: int = 5,         
    shuffle_facts: bool = True,        
    n_epoch: int = 80,
    lr: float = 5e-2,
    lambda_dh: float = 1e-4,    
    lambda_kl_ref: float = 0.0,      
    ref_prefix: int = 0,            
    ref_logits_cpu_external: Optional[List[torch.Tensor]] = None,
    vec_gate_init: float = 1.0,    
    vec_renorm: bool = True,
    vec_renorm_eps: float = 1e-8,
    save_vec_q: bool = False,
    gate_vec_u: bool = True,
    gate_vec_v: bool = True,
    forward_mode: str = "hook",          # {"hook","functional_call"}
    device: Optional[torch.device] = None,
    batch_id: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Spectral reverse optimization in top-r singular subspace (edited-model-only).
    """
    if device is None:
        device = next(edited_model.parameters()).device
    torch.set_printoptions(precision=5,sci_mode=False)
    assert len(prompts) == len(subjects), "prompts and subjects length mismatch"

    forward_mode = str(forward_mode).lower()
    assert forward_mode in {"hook", "functional_call"}, f"Unknown forward_mode: {forward_mode}"
    use_functional_call = (forward_mode == "functional_call")
    if use_functional_call:
        if _functional_call is None:
            raise RuntimeError("torch.func.functional_call is not available in this environment.")
        _base_params = {n: p.detach() for (n, p) in edited_model.named_parameters()}
        _base_buffers = {n: b for (n, b) in edited_model.named_buffers()}
    else:
        for p in edited_model.parameters():
            p.requires_grad_(False)
        _base_params = None
        _base_buffers = None

    # Decide layers
    if layers is None:
        layers = getattr(hparams, "layers", None)
        if layers is None:
            n_layer = getattr(getattr(edited_model, "config", None), "n_layer", None)
            if n_layer is None:
                n_layer = getattr(getattr(edited_model, "config", None), "num_hidden_layers", 0)
            layers = list(range(int(n_layer))) if n_layer else [0]


    flat_contexts, prompt_templates, template_subjects, actual_prompts = \
        _build_context_expanded_prompts(edited_model, tok, prompts, subjects)
    n_templates = len(flat_contexts)

    # Cache tokenized prompts and subject positions for optimization
    prompt_inputs = []
    subj_positions = []
    for p_str, s in zip(actual_prompts, template_subjects):
        enc = tok(p_str, return_tensors="pt")  # CPU tensors
        prompt_inputs.append(enc)
        pos = _find_last_subject_token_index(tok, p_str, s)
        subj_positions.append(pos)

    # Baseline logits/probs
    with torch.no_grad():
        baseline_logits_cpu = []          
        baseline_top1 = []                
        baseline_drop_topk_ids_cpu = []  
        drop_metric_topk = 10

        for enc in prompt_inputs:
            enc_d = {k: v.to(device) for k, v in enc.items()}
            out = edited_model(**enc_d)
            logits_cpu = out.logits[0, -1, :].detach().float().cpu()
            baseline_logits_cpu.append(logits_cpu.to(torch.float32))

            t1 = int(torch.argmax(logits_cpu).item())
            baseline_top1.append(t1)
            kd = min(drop_metric_topk, int(logits_cpu.numel()))
            baseline_drop_topk_ids_cpu.append(torch.topk(logits_cpu, k=kd, dim=-1).indices.cpu())
            # free GPU activations early
            del out, enc_d

    diag: Dict[str, Any] = {"per_layer": {}, "gates": {}, "meta": {}}

    # Optional pretrained reference distribution (for diagnostics).
    pre_log_probs_global = None
    if float(lambda_kl_ref) >= 0.0 and int(ref_prefix) > 0 and pre_rewrite_weights is not None:
        with torch.no_grad():
            pre_log_probs_global = []
            saved_w = snapshot_rewrite_weights(edited_model, hparams)
            load_rewrite_weights(edited_model, pre_rewrite_weights, device=device)
            try:
                for enc in prompt_inputs:
                    enc_d = {k: v.to(device) for k, v in enc.items()}
                    out_pre = edited_model(**enc_d)
                    logits_pre_cpu = out_pre.logits[0, -1, :].detach().float().cpu()
                    pre_log_probs_global.append(F.log_softmax(logits_pre_cpu, dim=-1).to(torch.float32))
                    del out_pre, enc_d
            finally:
                load_rewrite_weights(edited_model, saved_w, device=device)



    # ------------------------------------------------------------
    # Global coarse reference distribution
    # ------------------------------------------------------------
    ref_log_probs_global = None
    logits_ref_cpu_global = None
    if float(lambda_kl_ref) >= 0.0 and int(ref_prefix) > 0:
        if ref_logits_cpu_external is not None:
            assert len(ref_logits_cpu_external) == len(prompt_inputs), (
                f"ref_logits_cpu_external length mismatch: "
                f"{len(ref_logits_cpu_external)} vs expected {len(prompt_inputs)}"
            )
            logits_ref_cpu_global = [t.detach().cpu().to(torch.float32) for t in ref_logits_cpu_external]
            ref_log_probs_global = [F.log_softmax(t.float(), dim=-1).detach().cpu().to(torch.float32) for t in logits_ref_cpu_global]
        else:
            p = int(max(int(ref_prefix), 0))
            if p > 0:
                with torch.no_grad():
                    saved_w = snapshot_rewrite_weights(edited_model, hparams) 
                    for lj in layers:
                        try:
                            w_name_j = f"{hparams.rewrite_module_tmp.format(int(lj))}.weight"
                            Wp_j = nethook.get_parameter(edited_model, w_name_j)
                            mod_j = nethook.get_module(edited_model, hparams.rewrite_module_tmp.format(int(lj)))
                            if isinstance(mod_j, torch.nn.Linear):
                                d_k_j = int(Wp_j.shape[1])
                            else:
                                d_k_j = int(Wp_j.shape[0])
                            W_canon_j, preT_j = _weight_to_canon(Wp_j.detach(), d_k_j)
                            W_f_j = W_canon_j.float()
                            Uj, Sj, Vhj = torch.linalg.svd(W_f_j, full_matrices=False)
                            pj = int(min(p, Sj.numel()))
                            if pj <= 0:
                                continue
                            Sj_ref = Sj.clone()
                            Sj_ref[:pj] = 0.0
                            W_ref_j = (Uj * Sj_ref.unsqueeze(0)) @ Vhj
                            W_ref_param_j = _canon_to_param(W_ref_j.to(Wp_j.dtype), preT_j)
                            Wp_j.copy_(W_ref_param_j)
                        except Exception:
                            continue

                    # Compute reference logits/log-probs at prompt boundary for all prompts
                    logits_ref_cpu_global = []
                    ref_log_probs_global = []
                    for enc in prompt_inputs:
                        enc_d = {k: v.to(device) for k, v in enc.items()}
                        out_ref = edited_model(**enc_d)
                        logits_ref = out_ref.logits[0, -1, :].detach().float()
                        logits_ref_cpu_global.append(logits_ref.cpu().to(torch.float32))
                        ref_log_probs_global.append(F.log_softmax(logits_ref, dim=-1).detach().cpu().to(torch.float32))
                        del out_ref, enc_d

                    # Restore original (post) weights
                    load_rewrite_weights(edited_model, saved_w, device=device)

    for layer_idx in range(len(layers) - 1, -1, -1):
        layer = layers[layer_idx]
        prompt_keys_all, _ = compute_prompt_module_inout(
            edited_model,
            prompt_inputs,
            layer,
            hparams.rewrite_module_tmp,
            batch_size=key_batch_size,
        )
        subject_key_rows = []
        for token_keys, subj_pos in zip(prompt_keys_all, subj_positions):
            pos = int(subj_pos) if subj_pos is not None else int(token_keys.size(0) - 1)
            pos = max(0, min(pos, int(token_keys.size(0) - 1)))
            subject_key_rows.append(token_keys[pos])
        keys_all = torch.stack(subject_key_rows, dim=0)
        if keys_all.dim() != 2:
            raise RuntimeError(f"Expected keys_all dim=2 [n_pairs,d], got {tuple(keys_all.shape)}")
        K_flat = keys_all.contiguous()
        N, d_k = K_flat.shape

        # Weight and SVD
        w_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        W_param = nethook.get_parameter(edited_model, w_name)
        W_canon, pre_T = _weight_to_canon(W_param.detach(), d_k)  # [d_v, d_k]
        W_f = W_canon.float()
        U, S, Vh = torch.linalg.svd(W_f, full_matrices=False)
        r = min(rank, S.numel())
        if r <= 0:
            continue

        U_r = U[:, :r].to(device)              # [d_v, r]
        S_r = S[:r].to(device)                 # [r]
        V_r = Vh[:r, :].t().to(device)         # [d_k, r]

        # Key coords in right singular subspace
        K_dev = K_flat.to(device=device, dtype=torch.float32)
        A_unit = K_dev @ V_r                  # [N, r], row form = (V_r^T k)^T
        prompt_keys_dev = [k.to(device=device, dtype=torch.float32) for k in prompt_keys_all]
        prompt_A_unit = [k @ V_r for k in prompt_keys_dev]

        # Learnable parameters by mode
        assert gate_vec_u or gate_vec_v, "At least one of gate_vec_u / gate_vec_v must be enabled."
        params = []

        ku = U_r.size(0)
        kv = V_r.size(0)
        init_val = float(vec_gate_init)
        init_val = min(max(init_val, 1e-16), 1 - 1e-16)

        q_u = None
        q_v = None

        if gate_vec_u:
            q_u0 = torch.full(
                (r, ku),
                torch.logit(torch.tensor(init_val, dtype=torch.float32)),
                device=device,
            )
            q_u = torch.nn.Parameter(q_u0.clone())
            params.append(q_u)

        if gate_vec_v:
            q_v0 = torch.full(
                (r, kv),
                torch.logit(torch.tensor(init_val, dtype=torch.float32)),
                device=device,
            )
            q_v = torch.nn.Parameter(q_v0.clone())
            params.append(q_v)

        opt = torch.optim.Adam(params, lr=lr)
        mod = nethook.get_module(edited_model, hparams.rewrite_module_tmp.format(layer))

        best = {
            "loss": float("inf"),
            "stats": None,
            "q_u": None,
            "q_v": None,
        }

        # indices per component: [r, ku] and [r, kv]
        idx_u = torch.sort(U_r.abs().t(), dim=1, descending=True).indices if gate_vec_u else None  # [r, ku]
        idx_v = torch.sort(V_r.abs().t(), dim=1, descending=True).indices if gate_vec_v else None # [r, kv]

        # Precompute device copies for hook-free functional_call path
        W_f_dev = W_f.to(device)
        W_selected_orig_dev = (U_r * S_r.unsqueeze(0)) @ V_r.t()
        W_rest_dev = W_f_dev - W_selected_orig_dev

        def _build_W_new_canon(cur_param_local):
            """Return W_new (canonical [d_v, d_k]) on device, differentiable wrt cur_param_local."""
            U_mod, V_mod = cur_param_local
            W_top_new = (U_mod * S_r.unsqueeze(0)) @ V_mod.t()
            return W_top_new + W_rest_dev

        def _prompt_delta_norm(a_prompt_i, k_prompt_i, subj_pos, cur_param):
            U_mod, V_mod = cur_param
            y_old_all = (a_prompt_i * S_r.unsqueeze(0)) @ U_r.t()
            a_new_all = k_prompt_i @ V_mod
            y_new_all = (a_new_all * S_r.unsqueeze(0)) @ U_mod.t()
            delta_all = y_old_all - y_new_all
            h_edit_all = y_new_all
            if subj_pos is not None and delta_all.size(0) > 1:
                pos = int(subj_pos)
                if 0 <= pos < delta_all.size(0):
                    keep = torch.ones(delta_all.size(0), device=delta_all.device, dtype=torch.bool)
                    keep[pos] = False
                    delta_all = delta_all[keep]
                    h_edit_all = h_edit_all[keep]
            rel_mse = delta_all.pow(2).sum(dim=-1) / (h_edit_all.pow(2).sum(dim=-1) + vec_renorm_eps)
            return torch.sqrt(rel_mse.mean().clamp_min(0.0))

        def _forward_with_param(enc_inputs, subj_pos, a_i, k_i, a_prompt_i, k_prompt_i, cur_param):
            """Returns logits_new [V], delta_h [d_v], delta_h_norm scalar, reg_l1 scalar, reg_rot scalar."""
            U_mod, V_mod = cur_param
            coord_noise = torch.randn_like(k_i)
            coord_noise = coord_noise * (10 / coord_noise.norm().clamp_min(vec_renorm_eps))
            k_i_aug = k_i + coord_noise
            y_old = (U_r * S_r.unsqueeze(0)) @ a_i
            a_new = V_mod.t() @ k_i_aug
            y_new = (U_mod * S_r.unsqueeze(0)) @ a_new
            delta_h_hook = y_old - y_new
            prompt_dh_norm = _prompt_delta_norm(a_prompt_i, k_prompt_i, subj_pos, cur_param)

            if use_functional_call:
                W_new_canon = _build_W_new_canon(cur_param)
                W_new_param = _canon_to_param(W_new_canon, pre_T).to(dtype=W_param.dtype)
                params_call = dict(_base_params)
                params_call[w_name] = W_new_param
                out = _functional_call(edited_model, (params_call, _base_buffers), args=(), kwargs=enc_inputs)
                logits_new = out.logits[0, -1, :].float()
                delta_h = delta_h_hook
                return logits_new, delta_h, prompt_dh_norm

            delta_h = delta_h_hook
            delta_h_norm = prompt_dh_norm

            def hook_fn(module, inp, outp):
                if subj_pos is None:
                    return outp
                if isinstance(outp, tuple):
                    y0 = outp[0]
                    rest = outp[1:]
                else:
                    y0 = outp
                    rest = None
                y = y0
                if y.dim() == 3:
                    if subj_pos < y.size(1):
                        y = y.clone()
                        y[:, subj_pos, :] = y[:, subj_pos, :] - delta_h.to(y.dtype)
                elif y.dim() == 2:
                    if subj_pos < y.size(0):
                        y = y.clone()
                        y[subj_pos, :] = y[subj_pos, :] - delta_h.to(y.dtype)
                else:
                    return outp
                return (y, *rest) if rest is not None else y

            h = mod.register_forward_hook(lambda m, i, o: hook_fn(m, i, o))
            try:
                out = edited_model(**enc_inputs)
                logits_new = out.logits[0, -1, :].float()
            finally:
                h.remove()

            return logits_new, delta_h, delta_h_norm

        # Optimize
        ref_log_probs = ref_log_probs_global
        kl_pre_ref_mean = None

        # Diagnostics: how close coarse reference is to pretrained distribution (if both available)
        if ref_log_probs is not None and pre_log_probs_global is not None:
            with torch.no_grad():
                kls = []
                for i in range(N):
                    logp_pre = pre_log_probs_global[i].to(device=device, dtype=torch.float32)
                    logp_ref = ref_log_probs[i].to(device=device, dtype=torch.float32)
                    p_pre = torch.exp(logp_pre)
                    kls.append(torch.sum(p_pre * (logp_pre - logp_ref)).item())
                kl_pre_ref_mean = float(sum(kls) / max(len(kls), 1))

        for epoch in range(n_epoch):
            # Stats
            n_used = 0
            sum_drop = 0.0
            hit_count_top1 = 0
            hit_count_topk = 0

            # For logging: accumulate component-wise means (per-sample)
            epoch_loss_sum = 0.0
            epoch_loss_kl_ref = 0.0
            epoch_loss_dh = 0.0

            # Shuffle fact order each epoch (recommended)
            if shuffle_facts:
                perm = torch.randperm(N).tolist()
            else:
                perm = list(range(N))

            fact_batch_size_eff = int(max(1, fact_batch_size))
            num_batches = 0

            for bstart in range(0, N, fact_batch_size_eff):
                batch_ids = perm[bstart:bstart + fact_batch_size_eff]
                if len(batch_ids) == 0:
                    continue

                Mu = torch.ones((U_r.size(0), r), device=device, dtype=torch.float32)
                Mv = torch.ones((V_r.size(0), r), device=device, dtype=torch.float32)

                if gate_vec_u:
                    mu = torch.sigmoid(q_u)   # [r, ku]
                    Mu.scatter_(0, idx_u.t(), mu.t())
                else:
                    mu = None

                if gate_vec_v:
                    mv = torch.sigmoid(q_v)   # [r, kv]
                    Mv.scatter_(0, idx_v.t(), mv.t())
                else:
                    mv = None

                U_mod = U_r * Mu
                V_mod = V_r * Mv

                if vec_renorm:
                    U_mod = U_mod / U_mod.norm(dim=0, keepdim=True).clamp_min(vec_renorm_eps)
                    V_mod = V_mod / V_mod.norm(dim=0, keepdim=True).clamp_min(vec_renorm_eps)

                cur_param = (U_mod, V_mod)

                opt.zero_grad()
                batch_loss_sum = torch.zeros((), device=device)

                for ii, i in enumerate(batch_ids):
                    enc = {k: v.to(device) for k, v in prompt_inputs[i].items()}
                    subj_pos = subj_positions[i]
                    a_i = A_unit[i]
                    k_i = K_dev[i]
                    a_prompt_i = prompt_A_unit[i]
                    k_prompt_i = prompt_keys_dev[i]

                    drop_ids = baseline_drop_topk_ids_cpu[i].to(device)

                    logits_new, delta_h, dh_norm = _forward_with_param(
                        enc, subj_pos, a_i, k_i, a_prompt_i, k_prompt_i, cur_param
                    )

                    probs_new = torch.softmax(logits_new, dim=-1)

                    loss_kl_ref_i = torch.zeros((), device=device)
                    if ref_log_probs is not None:
                        logp_new = F.log_softmax(logits_new, dim=-1)
                        logp_ref = ref_log_probs[i].to(device=device, dtype=torch.float32)
                        p_ref = torch.exp(logp_ref)
                        loss_kl_ref_i = torch.sum(p_ref * (logp_ref - logp_new))

                    loss_dh_i = dh_norm * dh_norm

                    # Total per-fact loss
                    loss_i = (
                        + lambda_kl_ref * loss_kl_ref_i
                        + lambda_dh * loss_dh_i
                    )
                    batch_loss_sum = batch_loss_sum + loss_i

                    if verbose and (epoch % 100 == 0 or epoch == n_epoch - 1):
                        print(f"[layer {layer}] current prompt is: {tok.batch_decode(enc['input_ids'])}")
                        print(f"edited top k token={tok.batch_decode(torch.topk(torch.softmax(baseline_logits_cpu[i],dim=-1),k=20,dim=-1).indices)}")
                        print(f"edited top k prob ={torch.topk(torch.softmax(baseline_logits_cpu[i],dim=-1),k=20,dim=-1).values}")
                        print(f"reversed top k token={tok.batch_decode(torch.topk(probs_new.detach().cpu(),k=20,dim=-1).indices)}")
                        print(f"reversed top k prob ={torch.topk(probs_new.detach().cpu(),k=20,dim=-1).values}")
                        if pre_log_probs_global is not None:
                            pre_probs_i = torch.exp(pre_log_probs_global[i])
                            print(f"pretrained top k token={tok.batch_decode(torch.topk(pre_probs_i,k=20,dim=-1).indices)}")
                            print(f"pretrained top k prob ={torch.topk(pre_probs_i,k=20,dim=-1).values}")
                        if ref_log_probs is not None:
                            pr = torch.exp(ref_log_probs[i])
                            print(f"reference top k token={tok.batch_decode(torch.topk(pr,k=20,dim=-1).indices)}")
                            print(f"reference top k prob ={torch.topk(pr,k=20,dim=-1).values}")
                        print("*******************************")

                    with torch.no_grad():
                        dropped = int(torch.argmax(torch.softmax(baseline_logits_cpu[i],dim=-1) - probs_new.cpu()).item())
                        t1 = baseline_top1[i]
                        if dropped == t1:
                            hit_count_top1 += 1
                        if dropped in set(drop_ids.detach().cpu().tolist()):
                            hit_count_topk += 1
                        sum_drop += float((torch.softmax(baseline_logits_cpu[i],dim=-1)[t1] - probs_new[t1].cpu()).item())

                        epoch_loss_kl_ref += float((lambda_kl_ref * loss_kl_ref_i).item())
                        epoch_loss_dh += float((lambda_dh * loss_dh_i).item())
                    n_used += 1

                # Average data loss across minibatch
                batch_loss = batch_loss_sum / float(len(batch_ids))

                batch_loss.backward()
                opt.step()
                num_batches += 1
                epoch_loss_sum += float(batch_loss.item())

                with torch.no_grad():
                    if q_u is not None:
                        q_u.clamp_(-100.0, 100.0)
                    if q_v is not None:
                        q_v.clamp_(-100.0, 100.0)

            # End epoch: update best based on average batch loss
            with torch.no_grad():
                cur_loss = float(epoch_loss_sum / max(num_batches, 1))
                if cur_loss < best["loss"]:
                    best["loss"] = cur_loss
                    best["stats"] = {
                        "avg_top1_prob_drop": sum_drop / max(n_used, 1),
                        "confirm_hit_rate_top1": hit_count_top1 / max(n_used, 1),
                        "confirm_hit_rate_topk": hit_count_topk / max(n_used, 1),
                    }
                    best["q_u"] = q_u.detach().clone() if q_u is not None else None
                    best["q_v"] = q_v.detach().clone() if q_v is not None else None

            if verbose and (epoch % 10 == 0 or epoch == n_epoch - 1):
                msg = (
                    f"[spectral][layer {layer}] step={epoch} "
                    f"loss={cur_loss:.6f} "
                    f"loss_kl_ref={epoch_loss_kl_ref / max(n_used,1):.6f} "
                    f"loss_dh={epoch_loss_dh / max(n_used,1):.6f} "
                    f"avg_top1_prob_drop={sum_drop / max(n_used,1):.6f} "
                    f"hit_top1={hit_count_top1 / max(n_used,1):.3f} "
                    f"hit_topk={hit_count_topk / max(n_used,1):.3f} "
                )
                print(msg)
    
        # Build repaired weight and write back
        with torch.no_grad():
            q_u_use = best["q_u"] if best["q_u"] is not None else (q_u.detach() if q_u is not None else None)
            q_v_use = best["q_v"] if best["q_v"] is not None else (q_v.detach() if q_v is not None else None)

            Mu = torch.ones((U_r.size(0), r), device=device, dtype=torch.float32)
            Mv = torch.ones((V_r.size(0), r), device=device, dtype=torch.float32)

            mu = None
            mv = None

            if gate_vec_u:
                mu = torch.sigmoid(q_u_use).to(device)
                Mu.scatter_(0, idx_u.t(), mu.t())

            if gate_vec_v:
                mv = torch.sigmoid(q_v_use).to(device)
                Mv.scatter_(0, idx_v.t(), mv.t())

            U_top_new = (U_r * Mu)
            V_top_new = (V_r * Mv)

            if vec_renorm:
                U_top_new = U_top_new / U_top_new.norm(dim=0, keepdim=True).clamp_min(vec_renorm_eps)
                V_top_new = V_top_new / V_top_new.norm(dim=0, keepdim=True).clamp_min(vec_renorm_eps)

            U_top_new = U_top_new.cpu()
            V_top_new = V_top_new.cpu()
            Vh_top_new = V_top_new.t().contiguous()  # [r, d_k]

            S_r_cpu = S[:r].cpu()
            W_top_new = (U_top_new * S_r_cpu.unsqueeze(0)) @ Vh_top_new
            W_selected_orig = (U_r.cpu() * S_r_cpu.unsqueeze(0)) @ V_r.cpu().t()
            W_new = W_f.cpu() - W_selected_orig + W_top_new
            write_summary = {
                "mu_mean": float(mu.mean().item()) if mu is not None else 1.0,
                "mv_mean": float(mv.mean().item()) if mv is not None else 1.0,
            }

            W_new_param = _canon_to_param(W_new.to(W_param.dtype), pre_T)
            W_param.copy_(W_new_param)

        # Diagnostics
        layer_diag = {
            "rank_used": int(r),
            "best_loss": float(best["loss"]),
            "best_stats": best["stats"],
            "sv_top32_before": S[: min(32, len(S))].detach().cpu().float().tolist(),
            **write_summary,
        }
        if kl_pre_ref_mean is not None:
            layer_diag["kl_pre_ref_mean"] = kl_pre_ref_mean

        q_u_use = best["q_u"] if best["q_u"] is not None else (q_u.detach() if q_u is not None else None)
        q_v_use = best["q_v"] if best["q_v"] is not None else (q_v.detach() if q_v is not None else None)

        vec_q_path = None
        if bool(save_vec_q):
            try:
                import pickle
                vec_q_path = os.path.join(args.output_dir, "vector", edit_cache_id_full, spectral_gate_info_id, f"layer{int(layer)}_batch{int(batch_id)}__vec_q.pkl")
                os.makedirs(os.path.dirname(vec_q_path), exist_ok=True)
                payload = {
                    "layer": int(layer),
                    "batch_id": int(batch_id),
                    "gate_vec_u": bool(gate_vec_u),
                    "gate_vec_v": bool(gate_vec_v),
                    "q_u": q_u_use.detach().cpu() if q_u_use is not None else None,
                    "q_v": q_v_use.detach().cpu() if q_v_use is not None else None,
                }
                # Save idx_u/idx_v if available to map gates back to original dimensions.
                if 'idx_u' in locals() and idx_u is not None:
                    payload["idx_u"] = idx_u.detach().cpu()
                if 'idx_v' in locals() and idx_v is not None:
                    payload["idx_v"] = idx_v.detach().cpu()
                with open(vec_q_path, 'wb') as f:
                    pickle.dump(payload, f)
            except Exception as e:
                print(f"[vec_shrink][warn] failed to save q_u/q_v pickle for layer {layer}: {e}")

        with torch.no_grad():
            W_written, _ = _weight_to_canon(W_param.detach(), d_k)
            _, S_after, _ = torch.linalg.svd(W_written.float(), full_matrices=False)
            layer_diag["sv_top32_after"] = S_after[: min(32, len(S_after))].detach().cpu().float().tolist()
        diag["per_layer"][str(layer)] = layer_diag

    diag["meta"] = {
        "layers": layers,
        "rank": rank,
        "n_epoch": n_epoch,
        "lr": lr,
        "lambda_dh": lambda_dh,
        "n_items": len(prompts),
        "n_opt_datapoints": len(prompt_inputs),
        "n_templates": n_templates,
        "lambda_kl_ref": float(lambda_kl_ref),
        "ref_prefix": int(ref_prefix),
        "vec_gate_init": float(vec_gate_init),
        "vec_renorm": vec_renorm,
        "forward_mode": forward_mode,
        "gate_vec_u": bool(gate_vec_u),
        "gate_vec_v": bool(gate_vec_v),
    }
    return diag


def _run_method_reverse_baseline(
    *,
    editor,
    hparams,
    prompts: List[str],
    subjects: List[str],
    target_new: List[str],
    coarse_ref_logits: List[List[torch.Tensor]],
    case_ids: List[int],
    simIE: bool = False,
    lamHyper: float = 1.0,
    init_model: bool = False,
    solver: str = "LU",
):
    alg_name = getattr(editor, "alg_name", None)
    if alg_name not in {"ROME", "MEMIT", "AlphaEdit"}:
        return None
    _reset_alphaedit_cache_c()

    requests = []
    for prompt, subject, target, refs, case_id in zip(prompts, subjects, target_new, coarse_ref_logits, case_ids):
        ref_tensor = torch.stack([r.detach().cpu().float() for r in refs], dim=0) if isinstance(refs, list) else refs
        requests.append({
            "prompt": prompt,
            "subject": subject,
            "target_new": target if target else " ",
            "ground_truth": "<|endoftext|>",
            "case_id": int(case_id),
            "coarse_ref_logits": ref_tensor,
            "locality": {},
            "portability": {},
        })

    if alg_name == "ROME" and simIE:
        _, weights_copy = editor.apply_algo(
            editor.model,
            editor.tok,
            [requests[-1]],
            hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
        )
        simie = SimIE(lamHyper, init=init_model, solver=solver)
        simie.initializtion(editor.model_name, weights_copy, device=hparams.device)
        editor.model = simie.reset_parameter(editor.model)

        for request in requests:
            if simie.init:
                editor.model = simie.reset_parameter(editor.model)
            keys_cache = simie.cache(editor.model, [request], editor.tok)
            edited_model, _ = editor.apply_algo(
                editor.model,
                editor.tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=False,
            )
            editor.model = simie.update(edited_model, keys_cache)
    elif alg_name == "ROME":
        for request in requests:
            editor.apply_algo(
                editor.model,
                editor.tok,
                [request],
                hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=False,
            )
    else:
        editor.apply_algo(
            editor.model,
            editor.tok,
            requests,
            hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=False,
        )

    return {
        "alg_name": alg_name,
        "n_requests": int(len(requests)),
        "case_ids": [int(x) for x in case_ids],
        "target": "coarse_ref_distribution",
    }


def _run_alpha_reverse_baseline(
    *,
    editor,
    alpha_hparams: AlphaEditHyperParams,
    forward_hparams,
    prompts: List[str],
    subjects: List[str],
    target_new: List[str],
    coarse_ref_logits: List[List[torch.Tensor]],
    case_ids: List[int],
):
    """Reverse an arbitrary compatible forward edit with AlphaEdit.

    AlphaEdit normally edits a fixed, model-specific group of layers.  For a
    fair reverse baseline, use exactly the layers changed by the forward
    method instead.
    """
    _reset_alphaedit_cache_c()
    reverse_hparams = deepcopy(alpha_hparams)
    forward_layers = list(getattr(forward_hparams, "layers", []))
    if not forward_layers:
        return None

    default_layers = list(reverse_hparams.layers)
    reverse_hparams.layers = forward_layers
    reverse_hparams.layer_selection = "all"

    if forward_layers != default_layers:
        p_root, p_ext = os.path.splitext(reverse_hparams.P_loc)
        p_ext = p_ext or ".pt"
        layer_tag = "-".join(str(layer) for layer in forward_layers)
        reverse_hparams.P_loc = f"{p_root}__layers-{layer_tag}{p_ext}"

    requests = []
    for prompt, subject, target, refs, case_id in zip(
        prompts, subjects, target_new, coarse_ref_logits, case_ids
    ):
        ref_tensor = (
            torch.stack([r.detach().cpu().float() for r in refs], dim=0)
            if isinstance(refs, list)
            else refs.detach().cpu().float()
        )
        requests.append({
            "prompt": prompt,
            "subject": subject,
            "target_new": target if target else " ",
            "ground_truth": "<|endoftext|>",
            "case_id": int(case_id),
            "coarse_ref_logits": ref_tensor,
            "locality": {},
            "portability": {},
        })

    apply_AlphaEdit_to_model(
        editor.model,
        editor.tok,
        requests,
        reverse_hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=False,
    )

    return {
        "alg_name": "AlphaEdit",
        "forward_alg_name": getattr(editor, "alg_name", None),
        "layers": forward_layers,
        "projection_path": reverse_hparams.P_loc,
        "n_requests": int(len(requests)),
        "case_ids": [int(x) for x in case_ids],
        "target": "coarse_ref_distribution",
    }


def _reset_alphaedit_cache_c():
    """Make the next AlphaEdit call start without editing-history covariance.
    The results reported in the rebuttal were incorrect because this point was overlooked.
    """
    alphaedit_main.cache_c_new = False
    if hasattr(alphaedit_main, "cache_c"):
        delattr(alphaedit_main, "cache_c")


# ============================================================
# Step 4) Evaluation
# ============================================================

def _device_for_hparams(hparams) -> str:
    # The codebase uses cuda:{hparams.device}.
    try:
        return f"cuda:{hparams.device}"
    except Exception:
        return "cuda:0"

def _position_ids_from_attention_mask(attn: torch.Tensor) -> torch.Tensor:
    """
    attn: [B, L] (0=pad, 1=token)
    returns position_ids: [B, L]
      - pad positions get 0
      - real tokens get 0..len-1 (independent of left/right padding)
    """
    attn = attn.long()
    pos = attn.cumsum(dim=1) - 1              # pad -> -1, first token -> 0
    pos = pos.clamp(min=0)
    pos = pos * attn                          # force pad positions to 0
    return pos

@torch.no_grad()
def _batched_prompt_last_logits(model, tok, prompts: List[str], hparams, batch_size: int = 8):
    """Return logits at the last non-pad token for each prompt.

    Output: list of torch.Tensor [V] logits.
    """
    device = _device_for_hparams(hparams)
    before_padding = tok.padding_side
    tok.padding_side = 'left'
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=getattr(hparams, 'max_length', 512),
            return_tensors='pt',
        ).to(device)
        pos_ids = _position_ids_from_attention_mask(enc["attention_mask"]).to(enc["input_ids"].device)
        logits = model(**enc, position_ids=pos_ids).logits  # [B, L, V]

        # LEFT padding => last non-pad token is always at position L-1
        last_logits = logits[:, -1, :]  # [B, V]
        out.extend([t.detach().cpu().float() for t in last_logits])
    tok.padding_side = before_padding
    return out


@torch.no_grad()
def evaluate_masked_vs_baseline(
    *,
    model,
    baseline_weights: Optional[Dict[str, torch.Tensor]],
    masked_weights: Optional[Dict[str, torch.Tensor]],
    baseline_last_logits_cached: Optional[List[torch.Tensor]] = None,
    tok,
    hparams,
    prompts: List[str],
    targets: Optional[List[str]],
    eval_topk: int,
    eval_batch_size: int,
    eval_gen_len: int = 0,
    eval_group_name=None,
    cand_name=None,
    prompt_view_name=None,
):
    """
    Compare masked_model against baseline_model on a set of prompts.
    """
    if len(prompts) == 0:
        return {'n': 0}

    device = _device_for_hparams(hparams)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # ---- prompt-boundary logits----
    if baseline_last_logits_cached is not None:
        logits_base = baseline_last_logits_cached
    else:
        load_rewrite_weights(model, baseline_weights, device=device)
        logits_base = _batched_prompt_last_logits(model, tok, prompts, hparams, batch_size=eval_batch_size)

    load_rewrite_weights(model, masked_weights, device=device)
    logits_mask = _batched_prompt_last_logits(model, tok, prompts, hparams, batch_size=eval_batch_size)

    if cand_name != "post_edited" and eval_group_name == "remained":
        for logit_base, logit_mask, prompt in zip(logits_base,logits_mask,prompts):
            print(f"{prompt_view_name} remained set for {cand_name}: current prompt is: {prompt}")
            print(f"reversed model top k token={tok.batch_decode(torch.topk(torch.softmax(logit_mask, dim=-1),k=20,dim=-1).indices)}")
            print(f"reversed model top k prob ={torch.topk(torch.softmax(logit_mask, dim=-1),k=20,dim=-1).values}")
            print(f"baseline model top k token={tok.batch_decode(torch.topk(torch.softmax(logit_base, dim=-1),k=20,dim=-1).indices)}")
            print(f"baseline model top k prob ={torch.topk(torch.softmax(logit_base, dim=-1),k=20,dim=-1).values}")
            print("********************************")

    if cand_name != "post_edited" and eval_group_name == "reverse":
        for logit_base, logit_mask, prompt in zip(logits_base,logits_mask,prompts):
            print(f"{prompt_view_name} reverse set for {cand_name}: current prompt is: {prompt}")
            print(f"reversed model top k token={tok.batch_decode(torch.topk(torch.softmax(logit_mask, dim=-1),k=20,dim=-1).indices)}")
            print(f"reversed model top k prob ={torch.topk(torch.softmax(logit_mask, dim=-1),k=20,dim=-1).values}")
            print(f"baseline model top k token={tok.batch_decode(torch.topk(torch.softmax(logit_base, dim=-1),k=20,dim=-1).indices)}")
            print(f"baseline model top k prob ={torch.topk(torch.softmax(logit_base, dim=-1),k=20,dim=-1).values}")
            print("********************************")

    def _kl_evaluate(logits_p: torch.Tensor, logits_q: torch.Tensor, topk: int, eps: float = 1e-9):
        p = torch.softmax(logits_p, dim=-1)
        q = torch.softmax(logits_q, dim=-1)
        k = min(int(topk), int(p.numel()))
        idx_p = torch.topk(p, k=k, dim=-1).indices
        idx_q = torch.topk(q, k=k, dim=-1).indices
        idx = torch.unique(torch.cat([idx_p, idx_q], dim=0))
        p_u = p[idx]; q_u = q[idx]
        p_tail = torch.clamp(1.0 - p_u.sum(), min=0.0)
        q_tail = torch.clamp(1.0 - q_u.sum(), min=0.0)
        p_s = torch.cat([p_u, p_tail.unsqueeze(0)], dim=0) + eps
        q_s = torch.cat([q_u, q_tail.unsqueeze(0)], dim=0) + eps
        p_s = p_s / p_s.sum(); q_s = q_s / q_s.sum()
        kl_pq = float(torch.sum(p_s * (torch.log(p_s) - torch.log(q_s))).item())
        return kl_pq

    def _summ(v: List[float]):
        if len(v) == 0:
            return {'mean': float('nan'), 'p50': float('nan'), 'p95': float('nan')}
        arr = np.asarray(v, dtype=np.float64)
        return {'mean': float(np.mean(arr)), 'p50': float(np.percentile(arr, 50)), 'p95': float(np.percentile(arr, 95))}

    kl_bm= []
    agree_first = []
    # Top-k inclusion metrics (k=2):
    #  - base@1 in mask@2
    #  - mask@1 in base@2
    agree_first_base_in_mask_topk = []
    agree_first_mask_in_base_topk = []
    agree_topk = []
    t1_ids = None
    if targets is not None:
        t1_ids = []
        for t in targets:
            ids = tok.encode(' ' + t, add_special_tokens=False)
            t1_ids.append(ids[0] if len(ids) else None)

    for i in range(len(prompts)):
        lb = logits_base[i]; lm = logits_mask[i]
        tb = int(torch.argmax(lb).item()); tm = int(torch.argmax(lm).item())
        agree_first.append(1.0 if tb == tm else 0.0)

        k_number = 2
        k_number = min(k_number, int(lb.numel()))
        topk_b = torch.topk(lb, k=k_number, dim=-1).indices
        topk_m = torch.topk(lm, k=k_number, dim=-1).indices
        agree_first_base_in_mask_topk.append(1.0 if (tb in topk_m.tolist()) else 0.0)
        agree_first_mask_in_base_topk.append(1.0 if (tm in topk_b.tolist()) else 0.0)
        agree_topk.append(1.0 if (set(topk_m.tolist()) == set(topk_b.tolist())) else 0.0)

        kl_bm.append(_kl_evaluate(lb, lm, topk=eval_topk))

    prompt_boundary = {
        'n': len(prompts),
        'agree_first_token': float(np.mean(agree_first)),
        'agree_first_token_base_in_mask_top2': float(np.mean(agree_first_base_in_mask_topk)),
        'agree_first_token_mask_in_base_top2': float(np.mean(agree_first_mask_in_base_topk)),
        'agree_topk_token': float(np.mean(agree_topk)),
        'kl_base_to_mask_topk': _summ(kl_bm),
    }

    out = {
        'n': len(prompts),
        'prompt_boundary': prompt_boundary,
    }

    # ---- optional generation metrics ----
    if int(eval_gen_len) > 0:
        gen_len = int(eval_gen_len)

        def _pad_left(seqs: List[torch.Tensor], pad_id: int):
            max_len = max(int(s.numel()) for s in seqs)
            batch = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
            for i, s in enumerate(seqs):
                l = int(s.numel())
                batch[i, max_len - l:] = s
                attn[i, max_len - l:] = 1
            return batch, attn

        before_padding = tok.padding_side
        tok.padding_side = 'left'
        base_cont, mask_cont = [], []
        base_full_for_logits = []
        prompt_lens_actual = []
        forced_prefix_for_mask = []
        base_target_cont = []

        for i0 in range(0, len(prompts), eval_batch_size):
            batch_prompts = prompts[i0:i0+eval_batch_size]
            enc = tok(batch_prompts, padding=True, truncation=True, return_tensors='pt')
            input_ids = enc['input_ids']
            attn = enc['attention_mask']
            in_len = int(input_ids.size(1))
            input_ids_d = input_ids.to(device)
            attn_d = attn.to(device)

            # Baseline free generation uses gen_len + 1 so we can force the masked model's next token
            load_rewrite_weights(model, baseline_weights, device=device)
            gen_base = model.generate(
                input_ids=input_ids_d,
                attention_mask=attn_d,
                max_new_tokens=gen_len + 1,
                do_sample=False,
                pad_token_id=pad_id,
            ).detach().cpu()

            load_rewrite_weights(model, masked_weights, device=device)
            gen_mask = model.generate(
                input_ids=input_ids_d,
                attention_mask=attn_d,
                max_new_tokens=gen_len,
                do_sample=False,
                pad_token_id=pad_id,
            ).detach().cpu()

            for bi in range(gen_base.size(0)):
                prompt_len = int(attn[bi].sum().item())
                prompt_tokens = input_ids[bi, in_len - prompt_len:].cpu()
                prompt_lens_actual.append(prompt_len)

                cont_b_plus1 = gen_base[bi, in_len:][:gen_len + 1]
                cont_m = gen_mask[bi, in_len:][:gen_len]
                cont_b = cont_b_plus1[:gen_len]
                base_cont.append(cont_b)
                mask_cont.append(cont_m)
                base_full_for_logits.append(torch.cat([prompt_tokens, cont_b_plus1], dim=0))

                # For continuation metrics, force masked model's next token to baseline first token
                if cont_b_plus1.numel() >= 1:
                    forced_prefix_for_mask.append(torch.cat([prompt_tokens, cont_b_plus1[:1]], dim=0))
                    base_target_cont.append(cont_b_plus1[1:1+gen_len])
                else:
                    forced_prefix_for_mask.append(prompt_tokens.clone())
                    base_target_cont.append(torch.empty((0,), dtype=torch.long))

                if cand_name != "post_edited" and eval_group_name == "remained":
                    print(f"{prompt_view_name} remained set for {cand_name}: current prompt used for generate is: {tok.decode(prompt_tokens)}")
                    print(f"reversed model generate result={tok.decode(cont_m)}")
                    print(f"baseline model generate result=={tok.decode(cont_b)}")
                    print("********************************")

                if cand_name != "post_edited" and eval_group_name == "reverse":
                    print(f"{prompt_view_name} reverse set for {cand_name}: current prompt used for generate is: {tok.decode(prompt_tokens)}")
                    print(f"reversed model generate result={tok.decode(cont_m)}")
                    print(f"baseline model generate result=={tok.decode(cont_b)}")
                    print("********************************")

        tok.padding_side = before_padding

        eos_id = tok.eos_token_id if tok.eos_token_id is not None else pad_id
        exact_match, prefix_match_len = [], []
        for cb, cm in zip(base_cont, mask_cont):
            b = cb.tolist() + [eos_id] * max(0, gen_len - int(cb.numel()))
            m_ = cm.tolist() + [eos_id] * max(0, gen_len - int(cm.numel()))
            b = b[:gen_len]; m_ = m_[:gen_len]

            exact_match.append(1.0 if b == m_ else 0.0)
            pl = 0
            for t in range(gen_len):
                if b[t] == m_[t]:
                    pl += 1
                else:
                    break
            prefix_match_len.append(float(pl))

        out['gen_agreement'] = {
            'gen_len': gen_len,
            'exact_match_rate': float(np.mean(exact_match)) if len(exact_match) else float('nan'),
            'prefix_match_len': _summ(prefix_match_len),
            'prefix_match_frac': _summ([v / gen_len for v in prefix_match_len]) if len(prefix_match_len) else {'mean': float('nan'), 'p50': float('nan'), 'p95': float('nan')},
        }
        if cand_name == "post_edited":
            # Continuation metrics: force masked model's next token to baseline's next token,
            # then compare the following gen_len tokens / distributions.
            klbm_steps = []
            generate_continuation_match = []
            continuation_agree = []
            continuation_exact_half = []
            continuation_exact_full = []

            # First, free-run masked model conditioned on prompt + baseline next token
            forced_mask_cont = []
            tok.padding_side = 'left'
            for i0 in range(0, len(forced_prefix_for_mask), eval_batch_size):
                batch_forced = forced_prefix_for_mask[i0:i0+eval_batch_size]
                forced_ids, forced_attn = _pad_left(batch_forced, pad_id=pad_id)
                forced_ids_d = forced_ids.to(device)
                forced_attn_d = forced_attn.to(device)

                load_rewrite_weights(model, masked_weights, device=device)
                gen_mask_forced = model.generate(
                    input_ids=forced_ids_d,
                    attention_mask=forced_attn_d,
                    max_new_tokens=gen_len,
                    do_sample=False,
                    pad_token_id=pad_id,
                ).detach().cpu()

                in_len_forced = int(forced_ids.size(1))
                for bi in range(gen_mask_forced.size(0)):
                    cont_mf = gen_mask_forced[bi, in_len_forced:][:gen_len]
                    forced_mask_cont.append(cont_mf)

            tok.padding_side = before_padding

            # Token-level continuation agreement and exact-match under forced next token
            for cb_target, cmf in zip(base_target_cont, forced_mask_cont):
                b = cb_target.tolist() + [eos_id] * max(0, gen_len - int(cb_target.numel()))
                m_ = cmf.tolist() + [eos_id] * max(0, gen_len - int(cmf.numel()))
                b = b[:gen_len]; m_ = m_[:gen_len]
                generate_continuation_match.extend([1.0 if bt == mt else 0.0 for bt, mt in zip(b, m_)])
                half = max(1, gen_len // 2)
                continuation_exact_half.append(1.0 if b[:half] == m_[:half] else 0.0)
                continuation_exact_full.append(1.0 if b == m_ else 0.0)

            # Distribution similarity along baseline greedy prefixes after forcing first token.
            forced_full_for_logits = []
            forced_prompt_lens = []
            for prefix, target_cont in zip(forced_prefix_for_mask, base_target_cont):
                forced_full_for_logits.append(torch.cat([prefix, target_cont], dim=0))
                forced_prompt_lens.append(int(prefix.numel()))

            for i0 in range(0, len(forced_full_for_logits), eval_batch_size):
                batch_full = forced_full_for_logits[i0:i0+eval_batch_size]
                batch_prompt_lens = forced_prompt_lens[i0:i0+eval_batch_size]
                full_ids, full_attn = _pad_left(batch_full, pad_id=pad_id)
                full_ids_d = full_ids.to(device)
                full_attn_d = full_attn.to(device)
                pos_ids_full = _position_ids_from_attention_mask(full_attn_d).to(full_ids_d.device)

                load_rewrite_weights(model, baseline_weights, device=device)
                logits_b = model(full_ids_d, attention_mask=full_attn_d, position_ids=pos_ids_full).logits.detach().cpu()
                load_rewrite_weights(model, masked_weights, device=device)
                logits_m = model(full_ids_d, attention_mask=full_attn_d, position_ids=pos_ids_full).logits.detach().cpu()

                B, L, V = logits_b.shape
                for bi in range(B):
                    seq_len = int(full_attn[bi].sum().item())
                    offset = L - seq_len
                    prompt_len = int(batch_prompt_lens[bi])
                    cont_len = seq_len - prompt_len
                    steps = min(gen_len, cont_len)
                    if steps <= 0:
                        continue
                    for t in range(1, steps + 1):
                        # Now compare the continuation tokens AFTER the first forced token.
                        pos = offset + (prompt_len + t - 2)
                        lb = logits_b[bi, pos]
                        lm = logits_m[bi, pos]
                        tb = int(torch.argmax(lb).item())
                        tm = int(torch.argmax(lm).item())
                        continuation_agree.append(1.0 if tb == tm else 0.0)

                        klbm_steps.append(_kl_evaluate(lb, lm, topk=eval_topk))

            out['gen_distribution'] = {
                'gen_len': gen_len,
                'continuation_agree': float(np.mean(continuation_agree)) if len(continuation_agree) else float('nan'),
                'generate_continuation_match': float(np.mean(generate_continuation_match)) if len(generate_continuation_match) else float('nan'),
                'continuation_exact_match_rate_half': float(np.mean(continuation_exact_half)) if len(continuation_exact_half) else float('nan'),
                'continuation_exact_match_rate': float(np.mean(continuation_exact_full)) if len(continuation_exact_full) else float('nan'),
                'kl_base_to_mask_topk': _summ(klbm_steps),
            }

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', default="ROME", type=str)
    parser.add_argument('--model_name', default="llama-7b", type=str)
    parser.add_argument('--data_dir', default="./data", type=str)
    parser.add_argument('--data_type', default="ZsRE", type=str,
                        choices=['ZsRE', 'counterfact'])
    parser.add_argument('--output_dir', default='./outputs', type=str)

    # Dataset pool size
    parser.add_argument('--ds_size', default=1000, type=int,
                        help='Pool size to load from dataset files (first ds_size items).')

    # Split settings
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_edits', default=None, type=int,
                        help='How many items to sequentially edit (subset of ds_size). Default: ds_size')
    parser.add_argument('--reverse_k', default=1, type=int, help='How many items to query for reversal each time.')

    # Editing options
    parser.add_argument('--sequential_edit', default=True, action="store_true")
    parser.add_argument('--simIE', default=False, action="store_true")
    parser.add_argument('--lamHyper', default=1, type=float)
    parser.add_argument('--init_model', default=False, action="store_true")
    parser.add_argument('--solver', default='LU', type=str)
    parser.add_argument('--save_model', default=False, action="store_true")
    parser.add_argument('--edit_batch_size', default=1, type=int)

    # Reverse based on spectral gate optimization in top singular subspace
    parser.add_argument("--spectral_gate_rank", type=int, default=64)
    parser.add_argument("--spectral_gate_epoch", type=int, default=80)
    parser.add_argument("--spectral_gate_lr", type=float, default=5e-2)
    parser.add_argument("--spectral_gate_lambda_kl_ref", type=float, default=0.1)
    parser.add_argument("--spectral_gate_ref_prefix", type=int, default=64)
    parser.add_argument("--spectral_gate_lambda_dh", type=float, default=1e-4)
    parser.add_argument("--spectral_gate_vec_gate_init", type=float, default=1.0)
    parser.add_argument("--spectral_gate_vec_renorm", action="store_true")
    parser.add_argument("--spectral_gate_forward_mode", type=str, default="hook", choices=["hook", "functional_call"],)
    parser.add_argument("--spectral_gate_fact_batch_size", type=int, default=5)
    parser.add_argument("--spectral_gate_vec_u", action="store_true")
    parser.add_argument("--spectral_gate_vec_v", action="store_true")
    parser.add_argument("--spectral_save_vec_q", action="store_true")

    # Step 4) evaluation
    parser.add_argument('--do_eval', default=False, action='store_true',
                        help='Run Step 4 evaluation (build remained-only baseline and compare).')
    parser.add_argument('--eval_topk', type=int, default=100,
                        help='Top-k used for KL/JS computations on next-token distributions.')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Batch size for evaluation forward passes.')
    parser.add_argument('--eval_gen_len', type=int, default=10,
                        help='If >0, compute greedy-generation agreement and multi-step distribution similarity along baseline greedy prefixes (max_new_tokens). Set 0 to disable.')
    args = parser.parse_args()

    seed_everything(args.seed)

    # hparams path
    # PRUNE is implemented as "MEMIT + PRUNE(delta-reduction)" so we reuse MEMIT hparams.
    if args.editing_method == 'PRUNE':
        args.hparams_dir = f"hparams/MEMIT/{args.model_name}.yaml"
    else:
        args.hparams_dir = f"hparams/{args.editing_method}/{args.model_name}.yaml"
    args.data_dir = args.data_dir.replace('.', true_dir, 1)

    # select hparams class
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'PRUNE':
        editing_hparams = MEMITHyperParams
    else:
        raise NotImplementedError

    K = args.ds_size

    # -------------------------
    # Load dataset pool
    # -------------------------
    if args.data_type == 'ZsRE':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts_pool = [d['loc'] + ' ' + d['loc_ans'] for d in loc_data]

        prompts_pool = [d['src'] for d in edit_data]
        subject_pool = [d['subject'] for d in edit_data]
        rephrase_pool = [d['rephrase'] for d in edit_data]
        target_new_pool = [d['alt'] for d in edit_data]
        locality_prompts_pool = [d['loc'] for d in edit_data]
        locality_ans_pool = [d['loc_ans'] for d in edit_data]
        locality_inputs_pool = {
            'neighborhood': {
                'prompt': locality_prompts_pool,
                'ground_truth': locality_ans_pool
            },
        }

    elif args.data_type == 'counterfact':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/counterfact-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts_pool = [d['locality_prompt'] + ' ' + d['locality_ground_truth'] for d in loc_data]

        prompts_pool = [d['prompt'] for d in edit_data]
        subject_pool = [d['subject'] for d in edit_data]
        rephrase_pool = [d['rephrase_prompt'] for d in edit_data]
        target_new_pool = [d['target_new'] for d in edit_data]
        locality_prompts_pool = [d['locality_prompt'] for d in edit_data]
        locality_ans_pool = [d['locality_ground_truth'] for d in edit_data]
        locality_inputs_pool = {
            'neighborhood': {
                'prompt': locality_prompts_pool,
                'ground_truth': locality_ans_pool
            },
        }

    else:
        raise ValueError(f"Unknown data_type: {args.data_type}")
    
    if len(prompts_pool) < args.ds_size:
        print(f"ds_size {args.ds_size} set larger than real dataset size. Set ds_size to {len(prompts_pool)}")
        args.ds_size = len(prompts_pool)

    if args.num_edits is None or args.num_edits > args.ds_size:
        args.num_edits = args.ds_size

    rng = random.Random(args.seed)
    all_idx = list(range(len(prompts_pool)))
    edit_idx = rng.sample(all_idx, args.num_edits)

    # Build the sequential edit stream (E)
    prompts = _take(prompts_pool, edit_idx)
    subject = _take(subject_pool, edit_idx)
    rephrase_prompts = _take(rephrase_pool, edit_idx)
    target_new = _take(target_new_pool, edit_idx)
    loc_prompts = _take(loc_prompts_pool, edit_idx)
    locality_inputs = _slice_locality_inputs(locality_inputs_pool, edit_idx)

    # -------------------------
    # Load hparams
    # -------------------------
    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')
    hparams.model_name = hparams.model_name.replace('.', true_dir, 1)
    if hasattr(hparams, 'stats_dir'):
        hparams.stats_dir = hparams.stats_dir.replace('.', true_dir, 1)
    if hasattr(hparams, 'P_loc'):
        P_loc_model_name = hparams.model_name.split("/")[-1]
        if P_loc_model_name == "gpt-j-6b":
            P_loc_model_name = "EleutherAI_gpt-j-6B"
        hparams.P_loc = hparams.P_loc.replace('.', f'./data/stats/{P_loc_model_name}', 1)
        hparams.P_loc = hparams.P_loc.replace('.', true_dir, 1)
    if hasattr(hparams, 'archive'):
        hparams.archive = hparams.archive.replace('.', true_dir, 1)

    alpha_reverse_hparams = AlphaEditHyperParams.from_hparams(
        f'hparams/AlphaEdit/{args.model_name}.yaml'
    )
    alpha_reverse_hparams.model_name = alpha_reverse_hparams.model_name.replace('.', true_dir, 1)
    alpha_reverse_hparams.stats_dir = alpha_reverse_hparams.stats_dir.replace('.', true_dir, 1)
    alpha_p_model_name = alpha_reverse_hparams.model_name.split("/")[-1]
    if alpha_p_model_name == "gpt-j-6b":
        alpha_p_model_name = "EleutherAI_gpt-j-6B"
    alpha_reverse_hparams.P_loc = alpha_reverse_hparams.P_loc.replace(
        '.', f'./data/stats/{alpha_p_model_name}', 1
    )
    alpha_reverse_hparams.P_loc = alpha_reverse_hparams.P_loc.replace('.', true_dir, 1)

    # -------------------------
    # Output config
    # -------------------------
    edit_cache_id_full = (
        f'{hparams.model_name.split("/")[-1]}_{args.data_type}_{args.editing_method}'
        f'_Pool={args.ds_size}_Edits={args.num_edits}_Batch={args.edit_batch_size}'
        f'_Seed={args.seed}_{args.simIE}_{args.lamHyper}'
    )
    spectral_gate_info_id = (
        f'{args.spectral_gate_rank}_{args.spectral_gate_epoch}_{args.spectral_gate_lr}'
        f'_{args.spectral_gate_lambda_dh}_{args.spectral_gate_vec_gate_init}'
        f'_{"T" if args.spectral_gate_vec_renorm else "F"}_{args.spectral_gate_forward_mode[0]}_{args.spectral_gate_fact_batch_size}'
        f'_{"T" if args.spectral_gate_vec_u else "F"}_{"T" if args.spectral_gate_vec_v else "F"}'
        f'_{args.spectral_gate_lambda_kl_ref}_{args.spectral_gate_ref_prefix}_{args.reverse_k}'
    )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "model"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data_split"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "reverse_result", edit_cache_id_full), exist_ok=True)
    output_file = os.path.join(args.output_dir, "reverse_result", edit_cache_id_full, spectral_gate_info_id + '.json')

    print("See results at: ", output_file)

    run_config = {
        'data_type': args.data_type,
        'pool_size': args.ds_size,
        'num_edits': args.num_edits,
        'seed': args.seed,
        'model_name': hparams.model_name.split("/")[-1],
        'editing_method': args.editing_method,
        'simIE': args.simIE,
        'lamHyper': args.lamHyper,
        'init_model': args.init_model,
        'solver': args.solver,
        'output_dir': args.output_dir,
        # Stable run identifier for artifact filenames
        'run_id': edit_cache_id_full,
        'edit_idx': edit_idx,
        'other': hparams.to_dict(),
    }

    split_path = os.path.join(args.output_dir, "data_split", f"{edit_cache_id_full}__split.json")

    def _pack_rows(idxs: List[int]):
        rows = []
        for i in idxs:
            row = {
                'idx': int(i),
                'prompt': prompts_pool[i],
                'subject': subject_pool[i],
                'target_new': target_new_pool[i],
                'rephrase_prompt': (rephrase_pool[i] if rephrase_pool is not None else None),
                'loc_prompt': (loc_prompts_pool[i] if loc_prompts_pool is not None else None),
            }
            rows.append(row)
        return rows

    split_payload = {
        'run_id': edit_cache_id_full,
        'seed': int(args.seed),
        'pool_size': int(args.ds_size),
        'num_edits': int(args.num_edits),
        'edit_idx': edit_idx,
        'edited_examples': _pack_rows(edit_idx),
    }

    # Always write the split file so reruns are reproducible even if the run crashes later.
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split_payload, f, indent=2, ensure_ascii=False)
    run_config['split_path'] = split_path

    # -------------------------
    # Sequential editing (E)
    # -------------------------
    editor = BaseEditor.from_hparams(hparams)
    context_templates = get_context_templates(editor.model,editor.tok)
    if args.editing_method == 'PRUNE':
        reduce_name = 'log2' if "mistral" in args.model_name else 'log1_2'
        run_config['reduce_name'] = reduce_name
        editor.prune = PRUNE(reduce_name)

    # Snapshot full pretrained rewrite-layer weights (weight + optional bias) once.
    pre_rewrite_weights = None
    if hasattr(hparams, 'rewrite_module_tmp') and hasattr(hparams, 'layers'):
        try:
            pre_rewrite_weights = snapshot_rewrite_weights(editor.model, hparams)
        except Exception as e:
            print(f"[warn] Failed to snapshot pretrained rewrite weights: {e}")
            pre_rewrite_weights = None

    edited_layer_path = os.path.join(args.output_dir, "model", f"{edit_cache_id_full}__edited_layer.pth")

    if os.path.exists(edited_layer_path):
        print(f"[resume] Found cached edited layer: {edited_layer_path}")
        payload = torch.load(edited_layer_path, map_location="cpu")
        params = payload.get("params", payload)  # backward compatible

        edited_model = editor.model
        try:
            # Filter to plausible parameter entries (weights/biases). load_rewrite_weights will skip missing names.
            params_rewrite = {k: v for k, v in params.items() if isinstance(v, torch.Tensor) and (k.endswith('.weight') or k.endswith('.bias'))}
            if params_rewrite:
                load_rewrite_weights(edited_model, params_rewrite)
            else:
                # Backward compat: if payload isn't a flat dict, try load directly.
                if isinstance(params, dict):
                    load_rewrite_weights(edited_model, {k: v for k, v in params.items() if isinstance(v, torch.Tensor)})
        except Exception as e:
            print(f"[resume][warn] Failed to load cached weights into model: {e}")
        post_metrics = None
    else:
        if args.edit_batch_size == 1:
            post_metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                loc_prompts=loc_prompts,
                subject=subject,
                locality_inputs=locality_inputs,
                sequential_edit=args.sequential_edit,
                eval_metric='ppl' if args.data_type == 'hallucination' else 'token em',
                simIE=args.simIE,
                lamHyper=args.lamHyper,
                init_model=args.init_model,
                solver=args.solver,
                run_cfg=run_config,
                save_model=args.save_model,
            )
        else:
            post_metrics, edited_model, _ = editor.batch_edit(
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                loc_prompts=loc_prompts,
                subject=subject,
                locality_inputs=locality_inputs,
                sequential_edit=args.sequential_edit,
                eval_metric='ppl' if args.data_type == 'hallucination' else 'token em',
                simIE=args.simIE,
                lamHyper=args.lamHyper,
                init_model=args.init_model,
                solver=args.solver,
                run_cfg=run_config,
                save_model=args.save_model,
                edit_batch_size=args.edit_batch_size,
            )

    # Snapshot post-edit rewrite weights once. We'll re-load these weights to build multiple
    # "candidate" models (ours vs baselines) without keeping multiple full model copies.
    variant_weights: Dict[str, Dict[str, torch.Tensor]] = {}
    if hasattr(hparams, 'rewrite_module_tmp') and hasattr(hparams, 'layers'):
        variant_weights['post'] = snapshot_rewrite_weights(edited_model, hparams)

    batch_runs = []
    spectral_gate_diag = None
    eval_diag = None

    prompt_views = {
        'rephrase_prompt': rephrase_pool,
        'original_prompt': prompts_pool,
    }
    baseline_cache_full_edit = {}
    baseline_cache_pretrained = {}
    baseline_cache_reference = {}
    for prompt_view_name, prompt_pool_view in prompt_views.items():
        baseline_cache_full_edit[prompt_view_name] = {}
        baseline_cache_pretrained[prompt_view_name] = {}
        _prompts_all = _take(prompt_pool_view, edit_idx) if len(edit_idx) > 0 else []
        # Cache baseline prompt-boundary logits for FULL-EDIT and PRETRAINED once.
        # These references do not change across reverse batches, so we reuse them to save time.
        if args.do_eval and len(edit_idx) > 0:
            # --- full-edit baseline (post) ---
            if 'post' in variant_weights:
                load_rewrite_weights(edited_model, variant_weights['post'], device=_device_for_hparams(hparams))
            logits_full = _batched_prompt_last_logits(
                edited_model, editor.tok, _prompts_all, hparams, batch_size=int(args.eval_batch_size)
            )
            for idx_i, lg in zip(edit_idx, logits_full):
                baseline_cache_full_edit[prompt_view_name][int(idx_i)] = lg.detach().cpu()

            # --- pretrained baseline ---
            if pre_rewrite_weights is not None:
                load_rewrite_weights(edited_model, pre_rewrite_weights, device=_device_for_hparams(hparams))
            logits_pre = _batched_prompt_last_logits(
                edited_model, editor.tok, _prompts_all, hparams, batch_size=int(args.eval_batch_size)
            )
            for idx_i, lg in zip(edit_idx, logits_pre):
                baseline_cache_pretrained[prompt_view_name][int(idx_i)] = lg.detach().cpu()

            # restore post
            if 'post' in variant_weights:
                load_rewrite_weights(edited_model, variant_weights['post'], device=_device_for_hparams(hparams))

        # --- coarse reference baseline (truncate top singular values on ALL target layers simultaneously) ---
        # This reference is shared for both single-layer and multi-layer reverse, and is cached once per run.
        need_ref_cache = (
            int(getattr(args, "spectral_gate_ref_prefix", 0)) > 0
            and len(edit_idx) > 0
        )
        if need_ref_cache:
            p_ref = int(getattr(args, "spectral_gate_ref_prefix", 0))
            p_ref = max(0, p_ref)
            baseline_cache_reference[prompt_view_name] = {}

            # Apply truncation to all target layers (hparams.layers) simultaneously
            for lj in getattr(hparams, "layers", []):
                try:
                    w_name_j = f"{hparams.rewrite_module_tmp.format(int(lj))}.weight"
                    Wp_j = nethook.get_parameter(edited_model, w_name_j)
                    mod_j = nethook.get_module(edited_model, hparams.rewrite_module_tmp.format(int(lj)))
                    if isinstance(mod_j, torch.nn.Linear):
                        d_k_j = int(Wp_j.shape[1])
                    else:
                        d_k_j = int(Wp_j.shape[0])
                    W_canon_j, preT_j = _weight_to_canon(Wp_j.detach(), d_k_j)
                    W_f_j = W_canon_j.float()
                    Uj, Sj, Vhj = torch.linalg.svd(W_f_j, full_matrices=False)
                    pj = int(min(p_ref, Sj.numel()))
                    if pj <= 0:
                        continue
                    Sj_ref = Sj.clone()
                    Sj_ref[:pj] = 0.0
                    W_ref_j = (Uj * Sj_ref.unsqueeze(0)) @ Vhj
                    W_ref_param_j = _canon_to_param(W_ref_j.to(Wp_j.dtype), preT_j)
                    with torch.no_grad():
                        Wp_j.copy_(W_ref_param_j)
                except Exception:
                    continue

            # Compute reference logits for the EXACT context-expanded prompts used by reverse optimization
            cache_prompts = _take(prompt_pool_view, edit_idx)
            cache_subjects = _take(subject_pool, edit_idx)

            flat_contexts_cache, _, template_subjects_cache, actual_prompts_cache = \
                _build_context_expanded_prompts(edited_model, editor.tok, cache_prompts, cache_subjects)

            logits_ref = _batched_prompt_last_logits(
                edited_model,
                editor.tok,
                actual_prompts_cache,
                hparams,
                batch_size=int(args.eval_batch_size),
            )

            T = len(flat_contexts_cache)
            for j, idx_i in enumerate(edit_idx):
                start = j * T
                end = (j + 1) * T
                baseline_cache_reference[prompt_view_name][int(idx_i)] = [
                    lg.detach().cpu() for lg in logits_ref[start:end]
                ]

            # restore post
            if 'post' in variant_weights:
                load_rewrite_weights(edited_model, variant_weights['post'], device=_device_for_hparams(hparams))

    # Exhaustive reverse batching over edited set
    # -------------------------
    # We reverse k facts at a time, exhausting the entire edited set.
    # If --reverse_k is not provided, we treat --num_reverse as k for backward compatibility.
    k = args.reverse_k
    k = max(1, k)
    _rng_batches = random.Random(int(args.seed) + 13579)
    _edit_shuffled = list(edit_idx)
    _rng_batches.shuffle(_edit_shuffled)
    _edit_shuffled = _edit_shuffled[:(len(_edit_shuffled)//k)*k]
    reverse_batches = [_edit_shuffled[i:i+k] for i in range(0, len(_edit_shuffled), k)]


    # Iterate over exhaustive reverse batches
    for batch_id, reverse_idx in enumerate(reverse_batches):
        reverse_set = set(reverse_idx)
        remained_edit_idx = [i for i in edit_idx if i not in reverse_set]

        print("============================================================")
        print(f"[batch {batch_id+1}/{len(reverse_batches)}] reverse_k={len(reverse_idx)} | remained={len(remained_edit_idx)}")
        print("============================================================")

        # Ensure model is reset to full-edit state at the beginning of each batch
        if 'post' in variant_weights:
            load_rewrite_weights(edited_model, variant_weights['post'], device=_device_for_hparams(hparams))
        variant_weights.pop('method_reverse', None)
        variant_weights.pop('alpha_reverse', None)

        spectral_gate_diag = None
        method_reverse_diag = None
        alpha_reverse_diag = None
        if len(reverse_idx) == 0:
            print("[spectral_gate_reverse] empty reverse batch; skip.")
        else:
            reverse_prompts_paraphrase = [rephrase_pool[i] for i in reverse_idx]
            reverse_subjects = [subject_pool[i] for i in reverse_idx]
            ref_logits_cpu_external=(
                [
                    lg
                    for i in reverse_idx
                    for lg in baseline_cache_reference['rephrase_prompt'][int(i)]
                ]
                if len(baseline_cache_reference.get('rephrase_prompt', {})) else None
            )

            spectral_gate_diag = spectral_gate_reverse_optimize(
                edited_model=edited_model,
                pre_rewrite_weights=pre_rewrite_weights,
                tok=editor.tok,
                hparams=hparams,
                prompts=reverse_prompts_paraphrase,
                subjects=reverse_subjects,
                layers=hparams.layers,
                rank=args.spectral_gate_rank,
                n_epoch=args.spectral_gate_epoch,
                lr=args.spectral_gate_lr,
                lambda_dh=args.spectral_gate_lambda_dh,
                lambda_kl_ref=args.spectral_gate_lambda_kl_ref,
                ref_prefix=args.spectral_gate_ref_prefix,
                ref_logits_cpu_external=ref_logits_cpu_external,
                vec_gate_init=args.spectral_gate_vec_gate_init,
                vec_renorm=args.spectral_gate_vec_renorm,
                forward_mode=args.spectral_gate_forward_mode,
                fact_batch_size=args.spectral_gate_fact_batch_size,
                gate_vec_u=args.spectral_gate_vec_u,
                gate_vec_v=args.spectral_gate_vec_v,
                save_vec_q=args.spectral_save_vec_q,
                batch_id=batch_id,
                verbose=True,
            )
            if hasattr(hparams, 'rewrite_module_tmp') and hasattr(hparams, 'layers'):
                variant_weights['spectral_gate'] = snapshot_rewrite_weights(edited_model, hparams)

            method_ref_logits = [
                baseline_cache_reference['rephrase_prompt'][int(i)]
                for i in reverse_idx
            ] if len(baseline_cache_reference.get('rephrase_prompt', {})) else None
            if method_ref_logits is not None and hasattr(hparams, 'rewrite_module_tmp') and hasattr(hparams, 'layers'):
                print(f"[batch {batch_id}] building same-method reverse baseline")
                try:
                    load_rewrite_weights(edited_model, variant_weights['post'], device=_device_for_hparams(hparams))
                    method_reverse_diag = _run_method_reverse_baseline(
                        editor=editor,
                        hparams=hparams,
                        prompts=reverse_prompts_paraphrase,
                        subjects=reverse_subjects,
                        target_new=[target_new_pool[i] for i in reverse_idx],
                        coarse_ref_logits=method_ref_logits,
                        case_ids=reverse_idx,
                        simIE=args.simIE,
                        lamHyper=args.lamHyper,
                        init_model=args.init_model,
                        solver=args.solver,
                    )
                    if method_reverse_diag is not None:
                        variant_weights['method_reverse'] = snapshot_rewrite_weights(edited_model, hparams)
                except Exception as e:
                    method_reverse_diag = {'error': str(e)}
                    print(f"[method_reverse][warn] failed to build baseline: {e}")
                finally:
                    load_rewrite_weights(edited_model, variant_weights['post'], device=_device_for_hparams(hparams))

                print(f"[batch {batch_id}] building AlphaEdit reverse baseline")
                try:
                    alpha_reverse_diag = _run_alpha_reverse_baseline(
                        editor=editor,
                        alpha_hparams=alpha_reverse_hparams,
                        forward_hparams=hparams,
                        prompts=reverse_prompts_paraphrase,
                        subjects=reverse_subjects,
                        target_new=[target_new_pool[i] for i in reverse_idx],
                        coarse_ref_logits=method_ref_logits,
                        case_ids=reverse_idx,
                    )
                    if alpha_reverse_diag is not None:
                        variant_weights['alpha_reverse'] = snapshot_rewrite_weights(edited_model, hparams)
                except Exception as e:
                    alpha_reverse_diag = {'error': str(e)}
                    print(f"[alpha_reverse][warn] failed to build baseline: {e}")
                finally:
                    load_rewrite_weights(edited_model, variant_weights['post'], device=_device_for_hparams(hparams))

        # -------------------------
        # Evaluation for this batch
        # -------------------------
        eval_diag = None
        if args.do_eval:
            # Groups: remained vs full-edit, reverse vs pretrained
            # Evaluate on both rephrase prompts and original edit prompts.

            def _group(indices, prompt_pool):
                return _take(prompt_pool, indices), _take(target_new_pool, indices), _take(subject_pool, indices)

            eval_groups = {
                'remained': {'indices': remained_edit_idx},
                'reverse': {'indices': reverse_idx},
            }

            eval_diag = {
                'settings': {
                    'eval_topk': int(args.eval_topk),
                    'eval_batch_size': int(args.eval_batch_size),
                    'eval_gen_len': int(args.eval_gen_len),
                },
                'references': {
                    'full_edit': 'post_full_edit',
                    'pretrained': 'pretrained',
                    'group_policy': {
                        'remained': 'vs_full_edit',
                        'reverse': 'vs_pretrained',
                    },
                },
                'prompt_views': list(prompt_views.keys()),
                'candidates': {},
            }

            candidates = {
                'spectral_gate': variant_weights.get('spectral_gate', None),
                'method_reverse': variant_weights.get('method_reverse', None),
                'alpha_reverse': variant_weights.get('alpha_reverse', None),
                'post_edited': variant_weights.get('post', None),
            }
            candidates = {k: v for k, v in candidates.items() if v is not None}

            def _evaluate_cached_reference(reference_logits, baseline_logits):
                """Prompt-boundary metrics from aligned cached logits only."""
                agree_first, kl_bm = [], []
                eps = 1e-9
                for lm, lb in zip(reference_logits, baseline_logits):
                    agree_first.append(float(torch.argmax(lb) == torch.argmax(lm)))
                    p = torch.softmax(lb.float(), dim=-1)
                    q = torch.softmax(lm.float(), dim=-1)
                    k = min(int(args.eval_topk), int(p.numel()))
                    idx = torch.unique(torch.cat((torch.topk(p, k).indices, torch.topk(q, k).indices)))
                    p_u, q_u = p[idx], q[idx]
                    p_s = torch.cat((p_u, (1.0 - p_u.sum()).clamp_min(0).unsqueeze(0))) + eps
                    q_s = torch.cat((q_u, (1.0 - q_u.sum()).clamp_min(0).unsqueeze(0))) + eps
                    p_s, q_s = p_s / p_s.sum(), q_s / q_s.sum()
                    kl_bm.append(float(torch.sum(p_s * (torch.log(p_s) - torch.log(q_s)))))

                arr = np.asarray(kl_bm, dtype=np.float64)
                return {
                    'n': len(agree_first),
                    'prompt_boundary': {
                        'n': len(agree_first),
                        'agree_first_token': float(np.mean(agree_first)),
                        'kl_base_to_mask_topk': {
                            'mean': float(np.mean(arr)),
                            'p50': float(np.percentile(arr, 50)),
                            'p95': float(np.percentile(arr, 95)),
                        },
                    },
                }

            if any(baseline_cache_reference.get(view) for view in prompt_views):
                reference_diag = {
                    'comparisons': {
                        'rephrase_prompt': {'vs_full_edit': {}, 'vs_pretrained': {}},
                        'original_prompt': {'vs_full_edit': {}, 'vs_pretrained': {}},
                    }
                }
                for group_name, baseline_name, baseline_cache in (
                    ('remained', 'vs_full_edit', baseline_cache_full_edit),
                    ('reverse', 'vs_pretrained', baseline_cache_pretrained),
                ):
                    idxs = eval_groups[group_name]['indices']
                    if idxs:
                        for prompt_view_name in prompt_views:
                            reference_view = baseline_cache_reference.get(prompt_view_name, {})
                            baseline_view = baseline_cache.get(prompt_view_name, {})
                            if not all(int(i) in reference_view and int(i) in baseline_view for i in idxs):
                                continue
                            # Entry zero corresponds to the bare "{}" context,
                            # i.e. the requested original/rephrase prompt itself.
                            reference_logits = [reference_view[int(i)][0] for i in idxs]
                            baseline_logits = [baseline_view[int(i)] for i in idxs]
                            reference_diag['comparisons'][prompt_view_name][baseline_name][group_name] = \
                                _evaluate_cached_reference(reference_logits, baseline_logits)
                eval_diag['candidates']['reference'] = reference_diag

            for cand_name, cand_w in candidates.items():
                if cand_name != 'post_edited':
                    print("*****************************")
                    print(f"[batch {batch_id}] current evaluation candidate is {cand_name}")
                    print("*****************************")

                cand_diag = {
                    'comparisons': {
                        'rephrase_prompt': {'vs_full_edit': {}, 'vs_pretrained': {}},
                        'original_prompt': {'vs_full_edit': {}, 'vs_pretrained': {}},
                    }
                }

                for prompt_view_name, prompt_pool_view in prompt_views.items():
                    # 2) remained group: candidate vs full-edit reference
                    if variant_weights.get('post', None) is not None and 'remained' in eval_groups:
                        if cand_name != 'post_edited':
                            idxs = eval_groups['remained']['indices']
                            p_list, t_list, _ = _group(idxs, prompt_pool_view)
                            base_logits_cached = [baseline_cache_full_edit[prompt_view_name][int(i)] for i in idxs] if len(baseline_cache_full_edit) else None
                            cand_diag['comparisons'][prompt_view_name]['vs_full_edit']['remained'] = evaluate_masked_vs_baseline(
                                model=edited_model,
                                masked_weights=cand_w,
                                baseline_weights=variant_weights.get('post', None),
                                baseline_last_logits_cached=base_logits_cached,
                                tok=editor.tok,
                                hparams=hparams,
                                prompts=p_list,
                                targets=t_list,
                                eval_topk=int(args.eval_topk),
                                eval_batch_size=int(args.eval_batch_size),
                                eval_gen_len=int(args.eval_gen_len),
                                eval_group_name='remained',
                                cand_name=cand_name,
                                prompt_view_name=prompt_view_name,
                            )

                    # 3) reverse groups: candidate vs pretrained reference
                    if 'reverse' in eval_groups:
                        idxs = eval_groups['reverse']['indices']
                        if len(idxs) != 0:
                            p_list, t_list, _ = _group(idxs, prompt_pool_view)
                            base_logits_cached = [baseline_cache_pretrained[prompt_view_name][int(i)] for i in idxs] if len(baseline_cache_pretrained) else None
                            cand_diag['comparisons'][prompt_view_name]['vs_pretrained']['reverse'] = evaluate_masked_vs_baseline(
                                model=edited_model,
                                masked_weights=cand_w,
                                baseline_weights=pre_rewrite_weights,
                                baseline_last_logits_cached=base_logits_cached,
                                tok=editor.tok,
                                hparams=hparams,
                                prompts=p_list,
                                targets=t_list,
                                eval_topk=int(args.eval_topk),
                                eval_batch_size=int(args.eval_batch_size),
                                eval_gen_len=int(args.eval_gen_len),
                                eval_group_name='reverse',
                                cand_name=cand_name,
                                prompt_view_name=prompt_view_name,
                            )

                eval_diag['candidates'][cand_name] = cand_diag

        batch_runs.append({
            'batch_id': int(batch_id),
            'reverse_idx': [int(i) for i in reverse_idx],
            # 'remained_edit_idx': [int(i) for i in remained_edit_idx],
            'spectral_gate': spectral_gate_diag,
            'method_reverse': method_reverse_diag,
            'alpha_reverse': alpha_reverse_diag,
            'evaluation': eval_diag,
        })

    # End batches

    # Summarize across batches.
    batch_summary = {}
    try:
        def _new_bucket():
            return {
                'pb_agree_first': [],
                'pb_base_in_mask_top2': [],
                'pb_mask_in_base_top2': [],
                'pb_agree_topk': [],
                'pb_kl_bm': [],     
                'gen_exact_match_rate': [], 
                'gen_prefix_match_len': [],
                'gen_prefix_match_frac': [],   
                'gen_continuation_agree': [],       
                'gen_generate_continuation_match': [],
                'gen_cont_exact_half': [],
                'gen_cont_exact_full': [],
                'gen_kl_bm': [],
            }

        def _append_eval_metrics(bucket, eval_blob):
            if not isinstance(eval_blob, dict) or not eval_blob:
                return
            pb = (eval_blob.get('prompt_boundary') or {})
            gd = (eval_blob.get('gen_distribution') or {})
            ga = (eval_blob.get('gen_agreement') or {})

            def _append(dst_key, value):
                if value is None:
                    return
                try:
                    val = float(value)
                except Exception:
                    return
                if math.isnan(val):
                    return
                bucket[dst_key].append(val)

            _append('pb_agree_first', pb.get('agree_first_token'))
            _append('pb_base_in_mask_top2', pb.get('agree_first_token_base_in_mask_top2'))
            _append('pb_mask_in_base_top2', pb.get('agree_first_token_mask_in_base_top2'))
            _append('pb_agree_topk', pb.get('agree_topk_token'))
            if isinstance(pb.get('kl_base_to_mask_topk'), dict):
                _append('pb_kl_bm', pb['kl_base_to_mask_topk'].get('mean'))
            _append('gen_exact_match_rate', ga.get('exact_match_rate'))
            if isinstance(ga.get('prefix_match_len'), dict):
                _append('gen_prefix_match_len', ga['prefix_match_len'].get('mean'))
            if isinstance(ga.get('prefix_match_frac'), dict):
                _append('gen_prefix_match_frac', ga['prefix_match_frac'].get('mean'))
            _append('gen_continuation_agree', gd.get('continuation_agree'))
            _append('gen_generate_continuation_match', gd.get('generate_continuation_match'))
            _append('gen_cont_exact_half', gd.get('continuation_exact_match_rate_half'))
            _append('gen_cont_exact_full', gd.get('continuation_exact_match_rate'))
            if isinstance(gd.get('kl_base_to_mask_topk'), dict):
                _append('gen_kl_bm', gd['kl_base_to_mask_topk'].get('mean'))

        def _mean(xs):
            xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
            return float(sum(xs) / len(xs)) if len(xs) else float('nan')

        def _summarize_bucket(bucket):
            return {
                'n_points': {k: int(len(v)) for k, v in bucket.items()},
                'mean_pb_agree_first': _mean(bucket['pb_agree_first']),
                'mean_pb_agree_first_base_in_mask_top2': _mean(bucket['pb_base_in_mask_top2']),
                'mean_pb_agree_first_mask_in_base_top2': _mean(bucket['pb_mask_in_base_top2']),
                'mean_pb_agree_topk_token': _mean(bucket['pb_agree_topk']),
                'mean_pb_kl_base_to_mask_topk': _mean(bucket['pb_kl_bm']),
                'mean_gen_exact_match_rate': _mean(bucket['gen_exact_match_rate']),
                'mean_gen_prefix_match_len': _mean(bucket['gen_prefix_match_len']),
                'mean_gen_prefix_match_frac': _mean(bucket['gen_prefix_match_frac']),
                'mean_gen_continuation_agree': _mean(bucket['gen_continuation_agree']),
                'mean_gen_generate_continuation_match': _mean(bucket['gen_generate_continuation_match']),
                'mean_gen_continuation_exact_match_rate_half': _mean(bucket['gen_cont_exact_half']),
                'mean_gen_continuation_exact_match_rate': _mean(bucket['gen_cont_exact_full']),
                'mean_gen_kl_base_to_mask_topk': _mean(bucket['gen_kl_bm']),
            }

        def _summ_diff(higher_summary, lower_summary):
            """Return improvement values where positive means the first summary is better.

            For agreement / overlap metrics, larger is better => first - second.
            For divergence / distance metrics, smaller is better => second - first.
            """
            higher_is_better = {
                'mean_pb_agree_first',
                'mean_pb_agree_first_base_in_mask_top2',
                'mean_pb_agree_first_mask_in_base_top2',
                'mean_pb_agree_topk_token',
                'mean_gen_exact_match_rate',
                'mean_gen_prefix_match_len',
                'mean_gen_prefix_match_frac',
                'mean_gen_continuation_agree',
                'mean_gen_generate_continuation_match',
                'mean_gen_continuation_exact_match_rate_half',
                'mean_gen_continuation_exact_match_rate',
            }
            lower_is_better = {
                'mean_pb_kl_base_to_mask_topk',
                'mean_gen_kl_base_to_mask_topk',
            }
            out = {}
            for key in sorted(set(higher_is_better) | set(lower_is_better)):
                a = higher_summary.get(key)
                b = lower_summary.get(key)
                if a is None or b is None:
                    out[key] = float('nan')
                    continue
                try:
                    a = float(a)
                    b = float(b)
                except Exception:
                    out[key] = float('nan')
                    continue
                if math.isnan(a) or math.isnan(b):
                    out[key] = float('nan')
                elif key in higher_is_better:
                    out[key] = float(a - b)
                else:
                    out[key] = float(b - a)
            return out

        prompt_views = ['rephrase_prompt', 'original_prompt']
        summary_store = {
            pv: {
                'spectral_gate': {
                    'remained_vs_full_edit': _new_bucket(),
                    'reverse_vs_pretrained': _new_bucket(),
                },
                'post_edited': {
                    'reverse_vs_pretrained': _new_bucket(),
                },
                'method_reverse': {
                    'remained_vs_full_edit': _new_bucket(),
                    'reverse_vs_pretrained': _new_bucket(),
                },
                'alpha_reverse': {
                    'remained_vs_full_edit': _new_bucket(),
                    'reverse_vs_pretrained': _new_bucket(),
                },
                'reference': {
                    'remained_vs_full_edit': _new_bucket(),
                    'reverse_vs_pretrained': _new_bucket(),
                },
            }
            for pv in prompt_views
        }

        for br in batch_runs:
            ev = br.get('evaluation') or {}
            cands = (ev.get('candidates') or {})
            for pv in prompt_views:
                sg = cands.get('spectral_gate') or {}
                sg_comp = ((sg.get('comparisons') or {}).get(pv) or {})
                _append_eval_metrics(
                    summary_store[pv]['spectral_gate']['remained_vs_full_edit'],
                    ((sg_comp.get('vs_full_edit') or {}).get('remained') or {}),
                )
                _append_eval_metrics(
                    summary_store[pv]['spectral_gate']['reverse_vs_pretrained'],
                    ((sg_comp.get('vs_pretrained') or {}).get('reverse') or {}),
                )

                post = cands.get('post_edited') or {}
                post_comp = ((post.get('comparisons') or {}).get(pv) or {})
                _append_eval_metrics(
                    summary_store[pv]['post_edited']['reverse_vs_pretrained'],
                    ((post_comp.get('vs_pretrained') or {}).get('reverse') or {}),
                )

                method_rev = cands.get('method_reverse') or {}
                method_comp = ((method_rev.get('comparisons') or {}).get(pv) or {})
                _append_eval_metrics(
                    summary_store[pv]['method_reverse']['remained_vs_full_edit'],
                    ((method_comp.get('vs_full_edit') or {}).get('remained') or {}),
                )
                _append_eval_metrics(
                    summary_store[pv]['method_reverse']['reverse_vs_pretrained'],
                    ((method_comp.get('vs_pretrained') or {}).get('reverse') or {}),
                )

                alpha_rev = cands.get('alpha_reverse') or {}
                alpha_comp = ((alpha_rev.get('comparisons') or {}).get(pv) or {})
                _append_eval_metrics(
                    summary_store[pv]['alpha_reverse']['remained_vs_full_edit'],
                    ((alpha_comp.get('vs_full_edit') or {}).get('remained') or {}),
                )
                _append_eval_metrics(
                    summary_store[pv]['alpha_reverse']['reverse_vs_pretrained'],
                    ((alpha_comp.get('vs_pretrained') or {}).get('reverse') or {}),
                )

                reference = cands.get('reference') or {}
                reference_comp = ((reference.get('comparisons') or {}).get(pv) or {})
                _append_eval_metrics(
                    summary_store[pv]['reference']['remained_vs_full_edit'],
                    ((reference_comp.get('vs_full_edit') or {}).get('remained') or {}),
                )
                _append_eval_metrics(
                    summary_store[pv]['reference']['reverse_vs_pretrained'],
                    ((reference_comp.get('vs_pretrained') or {}).get('reverse') or {}),
                )

        batch_summary = {
            'n_batches': int(len(batch_runs)),
            'reverse_k': int(k),
        }
        for pv in prompt_views:
            sg_rem = _summarize_bucket(summary_store[pv]['spectral_gate']['remained_vs_full_edit'])
            sg_rev = _summarize_bucket(summary_store[pv]['spectral_gate']['reverse_vs_pretrained'])
            post_rev = _summarize_bucket(summary_store[pv]['post_edited']['reverse_vs_pretrained'])
            method_rem = _summarize_bucket(summary_store[pv]['method_reverse']['remained_vs_full_edit'])
            method_rev = _summarize_bucket(summary_store[pv]['method_reverse']['reverse_vs_pretrained'])
            alpha_rem = _summarize_bucket(summary_store[pv]['alpha_reverse']['remained_vs_full_edit'])
            alpha_rev = _summarize_bucket(summary_store[pv]['alpha_reverse']['reverse_vs_pretrained'])
            reference_rem = _summarize_bucket(summary_store[pv]['reference']['remained_vs_full_edit'])
            reference_rev = _summarize_bucket(summary_store[pv]['reference']['reverse_vs_pretrained'])
            improve_rev = _summ_diff(sg_rev, post_rev)
            improve_over_method = _summ_diff(sg_rev, method_rev)
            improve_over_alpha = _summ_diff(sg_rev, alpha_rev)
            batch_summary[pv] = {
                'spectral_gate': {
                    'remained_vs_full_edit': sg_rem,
                    'reverse_vs_pretrained': sg_rev,
                },
                'method_reverse': {
                    'remained_vs_full_edit': method_rem,
                    'reverse_vs_pretrained': method_rev,
                },
                'alpha_reverse': {
                    'remained_vs_full_edit': alpha_rem,
                    'reverse_vs_pretrained': alpha_rev,
                },
                'post_edited': {
                    'reverse_vs_pretrained': post_rev,
                },
                'reference': {
                    'remained_vs_full_edit': reference_rem,
                    'reverse_vs_pretrained': reference_rev,
                },
                'improvement_over_do_nothing_on_reverse_vs_pretrained': improve_rev,
                'improvement_over_method_reverse_on_reverse_vs_pretrained': improve_over_method,
                'improvement_over_alpha_reverse_on_reverse_vs_pretrained': improve_over_alpha,
            }
    except Exception as _e:
        batch_summary = {'error': str(_e)}

    # -------------------------
    # Write outputs
    # -------------------------
    out = {
        'run_config': run_config,
        'batch_summary': batch_summary,
        'batch_runs': batch_runs,
    }

    def _save_metrics_json(path, metrics_blob):
        if metrics_blob is None:
            return
        mean_metrics = summary_metrics(metrics_blob)
        payload = {
            "run_config": run_config,
            "metrics": metrics_blob,
            "mean_metrics": mean_metrics
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    post_metrics_path = os.path.join(args.output_dir, edit_cache_id_full + "__post_metrics.json")
    _save_metrics_json(post_metrics_path, post_metrics)

    with open(output_file, 'w') as f:
        json.dump(out, f, indent=4)
