#!/bin/bash

editing_method="MEMIT"
model_name="llama-7b"
data_type="ZsRE"
ds_size="1000"
lamHyper="10"
num_edits="100"
eval_topk="1000"
seed="45"

simIE=""
renorm="--spectral_gate_vec_renorm"
vec_u="--spectral_gate_vec_u"
vec_v="--spectral_gate_vec_v"

edit_batch_size="100"
spectral_gate_lr="5e-2"
spectral_gate_lambda_dh="10000"
spectral_gate_rank="512"
spectral_gate_lambda_kl_ref="10"
spectral_gate_ref_prefix="512"
spectral_gate_epoch="500"
spectral_gate_vec_gate_init="0.9"
reverse_k="50"

python main_spectral_reverse.py --editing_method $editing_method --model_name $model_name --data_type $data_type --ds_size $ds_size --edit_batch_size $edit_batch_size --lamHyper $lamHyper \
 --num_edits $num_edits --sequential_edit --save_model --do_eval --eval_topk $eval_topk --seed $seed\
 --spectral_gate_lr $spectral_gate_lr --spectral_gate_lambda_dh $spectral_gate_lambda_dh --spectral_gate_rank $spectral_gate_rank \
 --spectral_gate_lambda_kl_ref $spectral_gate_lambda_kl_ref --spectral_gate_ref_prefix $spectral_gate_ref_prefix --spectral_gate_epoch $spectral_gate_epoch --spectral_gate_vec_gate_init $spectral_gate_vec_gate_init --spectral_gate_forward_mode hook --reverse_k $reverse_k $simIE $renorm $vec_u $vec_v

