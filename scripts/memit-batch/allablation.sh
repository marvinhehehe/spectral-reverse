#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

#sh scripts/memit-batch/ablation_only_uside_cf_gpt2xl.sh > log/memit-batch_ablation_only_uside_cf_gpt2xl.log 2>&1
#sh scripts/memit-batch/ablation_only_uside_cf_llama3.sh > log/memit-batch_ablation_only_uside_cf_llama3.log 2>&1
#sh scripts/memit-batch/ablation_only_uside_zsre_gpt2xl.sh > log/memit-batch_ablation_only_uside_zsre_gpt2xl.log 2>&1
#sh scripts/memit-batch/ablation_only_uside_zsre_llama3.sh > log/memit-batch_ablation_only_uside_zsre_llama3.log 2>&1

#sh scripts/memit-batch/ablation_only_vside_cf_gpt2xl.sh > log/memit-batch_ablation_only_vside_cf_gpt2xl.log 2>&1
sh scripts/memit-batch/ablation_only_vside_cf_llama3.sh > log/memit-batch_ablation_only_vside_cf_llama3.log 2>&1
#sh scripts/memit-batch/ablation_only_vside_zsre_gpt2xl.sh > log/memit-batch_ablation_only_vside_zsre_gpt2xl.log 2>&1
sh scripts/memit-batch/ablation_only_vside_zsre_llama3.sh > log/memit-batch_ablation_only_vside_zsre_llama3.log 2>&1

#sh scripts/memit-batch/ablation_interven_cf_gpt2xl.sh > log/memit-batch_ablation_interven_cf_gpt2xl.log 2>&1
sh scripts/memit-batch/ablation_interven_cf_llama3.sh > log/memit-batch_ablation_interven_cf_llama3.log 2>&1
#sh scripts/memit-batch/ablation_interven_zsre_gpt2xl.sh > log/memit-batch_ablation_interven_zsre_gpt2xl.log 2>&1
sh scripts/memit-batch/ablation_interven_zsre_llama3.sh > log/memit-batch_ablation_interven_zsre_llama3.log 2>&1

#sh scripts/memit-batch/ablation_ref_cf_gpt2xl.sh > log/memit-batch_ablation_ref_cf_gpt2xl.log 2>&1
sh scripts/memit-batch/ablation_ref_cf_llama3.sh > log/memit-batch_ablation_ref_cf_llama3.log 2>&1
#sh scripts/memit-batch/ablation_ref_zsre_gpt2xl.sh > log/memit-batch_ablation_ref_zsre_gpt2xl.log 2>&1
sh scripts/memit-batch/ablation_ref_zsre_llama3.sh > log/memit-batch_ablation_ref_zsre_llama3.log 2>&1