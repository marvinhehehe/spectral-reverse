#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

sh scripts/rome/ablation_only_uside_cf_gpt2xl.sh > log/rome_ablation_only_uside_cf_gpt2xl.log 2>&1
sh scripts/rome/ablation_only_uside_cf_llama3.sh > log/rome_ablation_only_uside_cf_llama3.log 2>&1
sh scripts/rome/ablation_only_uside_cf_mistral.sh > log/rome_ablation_only_uside_cf_mistral.log 2>&1
sh scripts/rome/ablation_only_uside_zsre_gpt2xl.sh > log/rome_ablation_only_uside_zsre_gpt2xl.log 2>&1
sh scripts/rome/ablation_only_uside_zsre_llama3.sh > log/rome_ablation_only_uside_zsre_llama3.log 2>&1
sh scripts/rome/ablation_only_uside_zsre_mistral.sh > log/rome_ablation_only_uside_zsre_mistral.log 2>&1

sh scripts/rome/ablation_only_vside_cf_gpt2xl.sh > log/rome_ablation_only_vside_cf_gpt2xl.log 2>&1
sh scripts/rome/ablation_only_vside_cf_llama3.sh > log/rome_ablation_only_vside_cf_llama3.log 2>&1
sh scripts/rome/ablation_only_vside_cf_mistral.sh > log/rome_ablation_only_vside_cf_mistral.log 2>&1
sh scripts/rome/ablation_only_vside_zsre_gpt2xl.sh > log/rome_ablation_only_vside_zsre_gpt2xl.log 2>&1
sh scripts/rome/ablation_only_vside_zsre_llama3.sh > log/rome_ablation_only_vside_zsre_llama3.log 2>&1
sh scripts/rome/ablation_only_vside_zsre_mistral.sh > log/rome_ablation_only_vside_zsre_mistral.log 2>&1

#sh scripts/rome/ablation_interven_cf_gpt2xl.sh > log/rome_ablation_interven_cf_gpt2xl.log 2>&1
sh scripts/rome/ablation_interven_cf_llama3.sh > log/rome_ablation_interven_cf_llama3.log 2>&1
sh scripts/rome/ablation_interven_cf_mistral.sh > log/rome_ablation_interven_cf_mistral.log 2>&1
#sh scripts/rome/ablation_interven_zsre_gpt2xl.sh > log/rome_ablation_interven_zsre_gpt2xl.log 2>&1
sh scripts/rome/ablation_interven_zsre_llama3.sh > log/rome_ablation_interven_zsre_llama3.log 2>&1
sh scripts/rome/ablation_interven_zsre_mistral.sh > log/rome_ablation_interven_zsre_mistral.log 2>&1

#sh scripts/rome/ablation_ref_cf_gpt2xl.sh > log/rome_ablation_ref_cf_gpt2xl.log 2>&1
sh scripts/rome/ablation_ref_cf_llama3.sh > log/rome_ablation_ref_cf_llama3.log 2>&1
sh scripts/rome/ablation_ref_cf_mistral.sh > log/rome_ablation_ref_cf_mistral.log 2>&1
#sh scripts/rome/ablation_ref_zsre_gpt2xl.sh > log/rome_ablation_ref_zsre_gpt2xl.log 2>&1
sh scripts/rome/ablation_ref_zsre_llama3.sh > log/rome_ablation_ref_zsre_llama3.log 2>&1
sh scripts/rome/ablation_ref_zsre_mistral.sh > log/rome_ablation_ref_zsre_mistral.log 2>&1

#sh scripts/rome/ablation_aug_cf_gpt2xl.sh > log/rome_ablation_aug_cf_gpt2xl.log 2>&1
sh scripts/rome/ablation_aug_cf_llama3.sh > log/rome_ablation_aug_cf_llama3.log 2>&1
sh scripts/rome/ablation_aug_cf_mistral.sh > log/rome_ablation_aug_cf_mistral.log 2>&1
#sh scripts/rome/ablation_aug_zsre_gpt2xl.sh > log/rome_ablation_aug_zsre_gpt2xl.log 2>&1
sh scripts/rome/ablation_aug_zsre_llama3.sh > log/rome_ablation_aug_zsre_llama3.log 2>&1
sh scripts/rome/ablation_aug_zsre_mistral.sh > log/rome_ablation_aug_zsre_mistral.log 2>&1