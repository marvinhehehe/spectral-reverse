#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

sh scripts/rome-simie/ablation_only_uside_cf_gpt2xl.sh > log/rome-simie_ablation_only_uside_cf_gpt2xl.log 2>&1
sh scripts/rome-simie/ablation_only_uside_cf_llama3.sh > log/rome-simie_ablation_only_uside_cf_llama3.log 2>&1
sh scripts/rome-simie/ablation_only_uside_zsre_gpt2xl.sh > log/rome-simie_ablation_only_uside_zsre_gpt2xl.log 2>&1
sh scripts/rome-simie/ablation_only_uside_zsre_llama3.sh > log/rome-simie_ablation_only_uside_zsre_llama3.log 2>&1

sh scripts/rome-simie/ablation_only_vside_cf_gpt2xl.sh > log/rome-simie_ablation_only_vside_cf_gpt2xl.log 2>&1
sh scripts/rome-simie/ablation_only_vside_cf_llama3.sh > log/rome-simie_ablation_only_vside_cf_llama3.log 2>&1
sh scripts/rome-simie/ablation_only_vside_zsre_gpt2xl.sh > log/rome-simie_ablation_only_vside_zsre_gpt2xl.log 2>&1
sh scripts/rome-simie/ablation_only_vside_zsre_llama3.sh > log/rome-simie_ablation_only_vside_zsre_llama3.log 2>&1