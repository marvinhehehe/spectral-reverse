from datasets import load_metric, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import matthews_corrcoef, f1_score
from useful_functions import load_data, load_data_split, MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP
import math
import torch
import time
import numpy as np
import pickle

MAX_NUMBER_OF_FEW_SHOTS = 100

class COLAEval():
    def __init__(self, model, tokenizer, number_of_tests = None, number_of_few_shots = 0, eval_split = 'validation'):
        assert number_of_few_shots < MAX_NUMBER_OF_FEW_SHOTS, f"The number of few shots should not exceed {number_of_few_shots}"
        self.number_of_tests = number_of_tests
        self.number_of_few_shots = number_of_few_shots
        self.model = model
        self.tokenizer = tokenizer
        self.few_shots, self.eval_dataset = load_data_split('glue_eval/dataset/cola.pkl', number_of_few_shots, number_of_tests)
        self._initialize_prompts()


    def _initialize_prompts(self):
        self.prefix_prompt = 'Is this sentence linguistically acceptable?\n'

        self.postfix_prompt = 'Answer: '
        self.few_shot_context = []
        for _, few_shot in enumerate(self.few_shots):
            self.few_shot_context.append(f"{self.prefix_prompt}Sentence: {few_shot['sentence']}\nAnswer: {'No' if few_shot['label'] == 0 else 'Yes'}\n")

    
    # def _create_prompt(self, example):
    #     prompt = 'Sentence: ' + example['sentence'] + '\n'

    #     input_prompt = self.few_shot_context + self.prefix_prompt + prompt + self.postfix_prompt

    #     return input_prompt, example['sentence'], example['label']

    def _create_prompt(self, example, gen_len):
        prompt = 'Sentence: ' + example['sentence'] + '\n'
        question = self.prefix_prompt + prompt + self.postfix_prompt
        question_token_length = len(self.tokenizer(question)["input_ids"])
        remaining_token_length = MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP[self.model.config._name_or_path.lower().split('/')[-1]] - question_token_length - gen_len
        actual_few_shot = ""
        for few_shot in self.few_shot_context:
            few_shot_token_length = len(self.tokenizer(few_shot)["input_ids"])
            remaining_token_length -= few_shot_token_length
            if remaining_token_length < 0:
                break 
            actual_few_shot += few_shot
        input_prompt = actual_few_shot + question
        print(type(example['label']))
        return input_prompt, example['sentence'], example['label']


    def _get_answer(self, generated_text):
        answer_text = generated_text.split('Answer:')[-1].strip().strip()

        if 'yes' in answer_text.lower():
            return 1
        elif 'no' in answer_text.lower():
            return 0

        return -1

    def evaluate(self, gen_len = 10, print_logs = False):

        yes_tok, no_tok = (self.tokenizer(f" {n}")["input_ids"] for n in ['Yes', 'No'])

        if "llama" in self.model.config._name_or_path.lower():
            yes_tok = yes_tok[1:]
            no_tok = no_tok[1:]

        yes_len, no_len = (len(n) for n in [yes_tok, no_tok])

        suffixes = {0: ['Yes', yes_tok, yes_len], 1: ['No', no_tok, no_len]}

        correct = 0
        incorrect = 0
        invalid = 0


        pos_correct = 0
        neg_correct = 0
        pos_incorrect = 0
        neg_incorrect = 0

        predictions = []
        labels = []
        predictions_new = []
        stored_generations = []
        start = time.time()
        
        for s, example in enumerate(self.eval_dataset):
            
            input_prompt, sentence, label = self._create_prompt(example, gen_len)
            print(input_prompt)
            input_prompt_tok = self.tokenizer(input_prompt, return_tensors='pt').to('cuda')
            input_prompt_ids = input_prompt_tok["input_ids"]
            input_prompt_text = self.tokenizer.decode(input_prompt_ids[0], skip_special_tokens=True)

            prefix_tok_len = len(self.tokenizer(input_prompt)["input_ids"])

            if 'llama' in self.model.config._name_or_path.lower():
                prefix_tok_len = prefix_tok_len - 1

            max_len = input_prompt_ids.shape[1] + gen_len
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            output = self.model.generate(
                input_ids=input_prompt_ids,
                attention_mask=input_prompt_tok["attention_mask"],
                max_length=max_len,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answer = self._get_answer(generated_text)

            predictions.append(answer)
            labels.append(label)
        
            probs = [0 for _ in suffixes.keys()]
            gen_texts = [0 for _ in suffixes.keys()]

            for i in range(len(suffixes.keys())):
                prompt_tok = self.tokenizer([f"{input_prompt} {suffixes[i][0]}"], return_tensors="pt").to('cuda')

                with torch.no_grad():
                    logits = self.model(**prompt_tok).logits

                if 'llama' in self.model.config._name_or_path.lower():
                    logits = logits[:, 1:, :]

                cur_len = suffixes[i][2]

                for j in range(cur_len):
                    cur_tok = suffixes[i][1][j]
                    probs[i] += -torch.nn.functional.log_softmax(
                    logits[0, prefix_tok_len + j - 1, :], dim=0
                    )[cur_tok].item()
                probs[i] /= cur_len
                
                gen_texts[i] = self.tokenizer.decode(logits[0, prefix_tok_len - 1 : prefix_tok_len + cur_len - 1, :].argmax(dim = -1))

            prob_yes = np.exp(-probs[0])
            prob_no = np.exp(-probs[1])

            print(f"prob_yes: {prob_yes}, prob_no: {prob_no}")
            answer_new = 1 if prob_yes > prob_no else 0
            predictions_new.append(answer_new)

            if answer == -1:
                invalid += 1
            else:

                if answer == label:
                    correct += 1

                    if label == 1:
                        pos_correct += 1
                    elif label == 0:
                        neg_correct += 1

                else:
                    incorrect += 1

                    if label == 1:
                        pos_incorrect += 1
                    elif label == 0:
                        neg_incorrect += 1


            exp_temp_dict = {
                'sentence': sentence,
                'input_prompt': input_prompt_text,
                'true_answer': 'Yes' if label == 1 else 'No',
                'generated_text': generated_text.replace(input_prompt_text, ''),
                'answer': answer,
                'correct': answer == label,
                'prob_yes': prob_yes,
                'prob_no': prob_no,
                'highest_probability_answer': 'Yes' if answer_new == 1 else 'No',
                'correct_new': answer_new == label,
            }
            stored_generations.append(exp_temp_dict)
            if print_logs:
                mcc = matthews_corrcoef(labels, predictions)
                f1 = f1_score(labels, predictions, average='weighted')
                print(generated_text)
                print(correct, incorrect, invalid, s+1, '|', pos_correct, neg_correct, '|', pos_incorrect, neg_incorrect, '|ACC: ', correct / (correct + incorrect + invalid), '|MCC:', mcc, '|F1:', f1)
                print('--'*50)

        end = time.time()
        mcc = matthews_corrcoef(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        f1_new = f1_score(labels, predictions_new, average='weighted')
        result_dict = {
            'correct': correct,
            'incorrect': incorrect,
            'invalid': invalid,
            'total': s+1,
            'f1': f1,
            'f1_new': f1_new,
            'mcc': mcc,
            'time': end-start,
        }
        return result_dict, stored_generations

# ===========================
# Spectral edit/reverse loaders
# ===========================

def _load_cached_params_pth(pth_path: str) -> dict:
    """Load a saved edited-layer checkpoint and return a dict name->tensor (CPU)."""
    payload = torch.load(pth_path, map_location="cpu")
    if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], dict):
        return payload["params"]
    # backward compat: the whole file is a dict of params
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unrecognized checkpoint format at {pth_path}")

@torch.no_grad()
def _apply_params_to_model_(model, params_cpu: dict, device: str = "cuda", strict: bool = False) -> None:
    """Copy a (partial) state dict into model parameters. Only keys present in params_cpu are applied."""
    missing = []
    for name, t_cpu in params_cpu.items():
        if not isinstance(t_cpu, torch.Tensor):
            continue
        try:
            p = dict(model.named_parameters())[name]
        except KeyError:
            if strict:
                missing.append(name)
            continue
        p.copy_(t_cpu.to(device=p.device, dtype=p.dtype))
    if strict and missing:
        raise KeyError(f"Missing {len(missing)} params in model, e.g. {missing[:5]}")

def _format_layer_path(template: str, layer: int) -> str:
    # supports templates containing '{int(layer)}' or '{layer}'
    if "{int(layer)}" in template:
        return template.replace("{int(layer)}", str(int(layer)))
    if "{layer}" in template:
        return template.format(layer=int(layer))
    return template  # assume already concrete

def _resolve_llama_downproj_name(model, layer: int) -> str:
    """Try to resolve a LLaMA MLP down_proj weight name for a given layer."""
    # Most HF LLaMA checkpoints use 'model.layers.{i}.mlp.down_proj.weight'
    candidates = [
        f"model.layers.{layer}.mlp.down_proj.weight",
        f"model.model.layers.{layer}.mlp.down_proj.weight",  # some wrappers
    ]
    sd_keys = set(model.state_dict().keys())
    for c in candidates:
        if c in sd_keys:
            return c
    # fallback: search by substring
    pat = f"layers.{layer}.mlp.down_proj.weight"
    for k in sd_keys:
        if k.endswith(pat) or (f"layers.{layer}." in k and k.endswith("mlp.down_proj.weight")):
            return k
    raise KeyError(f"Could not resolve down_proj weight name for layer {layer}")

@torch.no_grad()
def apply_vec_shrink_reverse_inplace(
    model,
    *,
    vec_q_pkl_template: str,
    layers: list,
    device: str = "cuda",
    weight_name_resolver=None,
    renorm_eps: float = 1e-8,
) -> None:
    """Apply vec_shrink reversal (from saved q_u/q_v pickles) to the model in-place.

    Assumes the pickle payload contains (at least):
      - q_u: [r, k_u], q_v: [r, k_v]
      - idx_u: [r, k_u] (optional; if missing we recompute top-|.| indices from current SVD U)
      - idx_v: [r, k_v] (optional; if missing we recompute from current SVD V)
      - vec_topk_u, vec_topk_v
    IMPORTANT: no clamping/saturation is applied (per your instruction).
    """
    model.eval()
    if weight_name_resolver is None:
        weight_name_resolver = _resolve_llama_downproj_name

    for layer in layers:
        pkl_path = _format_layer_path(vec_q_pkl_template, layer)
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        q_u = payload["q_u"].to(device=device, dtype=torch.float32)
        q_v = payload["q_v"].to(device=device, dtype=torch.float32)
        mu = torch.sigmoid(q_u)   # NO CLAMP
        mv = torch.sigmoid(q_v)

        r = int(q_u.shape[0])
        k_u = int(q_u.shape[1])
        k_v = int(q_v.shape[1])

        # Resolve the edited site weight
        w_name = weight_name_resolver(model, int(layer))
        W = model.state_dict()[w_name].to(device=device, dtype=torch.float32)  # [out,in]

        # SVD of current (edited) weight
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r_use = min(r, S.numel())
        U_r = U[:, :r_use].contiguous()
        V_r = Vh[:r_use, :].t().contiguous()
        S_r = S[:r_use].contiguous()

        # idx_u/idx_v either from payload or computed from current vectors
        idx_u = payload.get("idx_u", None)
        idx_v = payload.get("idx_v", None)

        if idx_u is None:
            # top-|.| indices per column
            idx_u = torch.topk(U_r.abs(), k=min(k_u, U_r.shape[0]), dim=0).indices.t().contiguous()  # [r,k]
        else:
            idx_u = idx_u.to(device=device, dtype=torch.long)
        if idx_v is None:
            idx_v = torch.topk(V_r.abs(), k=min(k_v, V_r.shape[0]), dim=0).indices.t().contiguous()
        else:
            idx_v = idx_v.to(device=device, dtype=torch.long)

        # Match shapes if r_use < r (rare)
        idx_u = idx_u[:r_use, :min(idx_u.shape[1], k_u)]
        idx_v = idx_v[:r_use, :min(idx_v.shape[1], k_v)]
        mu = mu[:r_use, :idx_u.shape[1]]
        mv = mv[:r_use, :idx_v.shape[1]]

        # Apply entry-wise gates to U_r and V_r (scatter multiply)
        U_mod = U_r.clone()
        V_mod = V_r.clone()

        # U updates
        cols_u = torch.arange(r_use, device=device).unsqueeze(1).expand(r_use, idx_u.shape[1]).reshape(-1)
        rows_u = idx_u.reshape(-1)
        U_mod[rows_u, cols_u] *= mu.reshape(-1)

        # V updates
        cols_v = torch.arange(r_use, device=device).unsqueeze(1).expand(r_use, idx_v.shape[1]).reshape(-1)
        rows_v = idx_v.reshape(-1)
        V_mod[rows_v, cols_v] *= mv.reshape(-1)

        # Renormalize each column back to original norm (SVD vectors are unit norm)
        U_mod = U_mod / U_mod.norm(dim=0, keepdim=True).clamp_min(renorm_eps)
        V_mod = V_mod / V_mod.norm(dim=0, keepdim=True).clamp_min(renorm_eps)

        # Reconstruct: keep complement unchanged
        W_top_old = (U_r * S_r.unsqueeze(0)) @ V_r.t()
        W_top_new = (U_mod * S_r.unsqueeze(0)) @ V_mod.t()
        W_new = W - W_top_old + W_top_new

        # Write back
        # Convert to model dtype/device
        W_new = W_new.to(dtype=model.state_dict()[w_name].dtype)
        # Use param copy to avoid replacing tensors in state_dict
        param = dict(model.named_parameters()).get(w_name, None)
        if param is None:
            # Some models put weights in buffers? unlikely; fallback via state_dict load
            tmp = model.state_dict()
            tmp[w_name] = W_new.detach().cpu()
            model.load_state_dict(tmp, strict=False)
        else:
            param.copy_(W_new.to(device=param.device))

def load_edited_model_inplace(model, edited_layer_pth: str, device: str = "cuda") -> None:
    """Load the saved edited-layer weights into model (in-place)."""
    params = _load_cached_params_pth(edited_layer_pth)
    _apply_params_to_model_(model, params, device=device, strict=False)

if __name__ == '__main__':
    # ===== User configuration (edit these paths) =====
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    edited_layer_pth = './outputs/model/Meta-Llama-3-8B-Instruct_counterfact_MEMIT_Pool=1000_Edits=100_Batch=100_Seed=45_False_10.0__edited_layer.pth'

    # First-batch vec_shrink gates (batch0000) for layers 4-8
    vec_q_pkl_template = './outputs/vector/Meta-Llama-3-8B-Instruct_counterfact_MEMIT_Pool=1000_Edits=100_Batch=100_Seed=45_False_10.0/512_500_0.05_100.0_0.9_T_h_5_T_T_10.0_512_50/layer{int(layer)}__vec_q.pkl'
    layers = [4, 5, 6, 7, 8]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ===== Load base model =====
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # ===== Load edited weights into model =====
    print(f"[load] applying edited-layer weights: {edited_layer_pth}")
    load_edited_model_inplace(model, edited_layer_pth, device=device)

    # ===== Evaluate edited model =====
    print("[eval] COLA on EDITED model for ZsRE")
    cola_eval = COLAEval(model, tokenizer)
    edited_result, _ = cola_eval.evaluate(print_logs=False)
    print("[result][edited]", edited_result)

    # ===== Apply vec_shrink reversal (first batch only) =====
    print(f"[reverse] applying vec_shrink gates from template: {vec_q_pkl_template}")
    apply_vec_shrink_reverse_inplace(
        model,
        vec_q_pkl_template=vec_q_pkl_template,
        layers=layers,
        device=device,
    )

    # ===== Evaluate reversed model =====
    print("[eval] COLA on REVERSED model for ZsRE")
    cola_eval = COLAEval(model, tokenizer)
    reversed_result, _ = cola_eval.evaluate(print_logs=False)
    print("[result][reversed]", reversed_result)

    # ===== Evaluate Counterfact part ===
    edited_layer_pth = './outputs/model/Meta-Llama-3-8B-Instruct_ZsRE_MEMIT_Pool=1000_Edits=100_Batch=100_Seed=45_False_10.0__edited_layer.pth'
    # First-batch vec_shrink gates (batch0000) for layers 4-8
    vec_q_pkl_template = './outputs/vector/Meta-Llama-3-8B-Instruct_ZsRE_MEMIT_Pool=1000_Edits=100_Batch=100_Seed=45_False_10.0/512_500_0.05_10000.0_0.9_T_h_5_T_T_10.0_512_50/layer{int(layer)}__vec_q.pkl'
    layers = [4, 5, 6, 7, 8]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ===== Load edited weights into model =====
    print(f"[load] applying edited-layer weights: {edited_layer_pth}")
    load_edited_model_inplace(model, edited_layer_pth, device=device)

    # ===== Evaluate edited model =====
    print("[eval] COLA on EDITED model for counterfact")
    cola_eval = COLAEval(model, tokenizer)
    edited_result, _ = cola_eval.evaluate(print_logs=False)
    print("[result][edited]", edited_result)

    # ===== Apply vec_shrink reversal (first batch only) =====
    print(f"[reverse] applying vec_shrink gates from template: {vec_q_pkl_template}")
    apply_vec_shrink_reverse_inplace(
        model,
        vec_q_pkl_template=vec_q_pkl_template,
        layers=layers,
        device=device,
    )

    # ===== Evaluate reversed model =====
    print("[eval] COLA on REVERSED model for counterfact")
    cola_eval = COLAEval(model, tokenizer)
    reversed_result, _ = cola_eval.evaluate(print_logs=False)
    print("[result][reversed]", reversed_result)
