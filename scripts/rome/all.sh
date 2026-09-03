#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

sh scripts/rome/cf_mistral_100_1_seed45.sh > log/rome_cf_mistral_100_1_seed45.log 2>&1
sh scripts/rome/cf_llama3_100_1_seed45.sh > log/rome_cf_llama3_100_1_seed45.log 2>&1
sh scripts/rome/cf_gpt2xl_100_1_seed45.sh > log/rome_cf_gpt2xl_100_1_seed45.log 2>&1
sh scripts/rome/cf_gptj_100_1_seed45.sh > log/rome_cf_gptj_100_1_seed45.log 2>&1
sh scripts/rome/cf_llama2_100_1_seed45.sh > log/rome_cf_llama2_100_1_seed45.log 2>&1
sh scripts/rome/zsre_mistral_100_1_seed45.sh > log/rome_zsre_mistral_100_1_seed45.log 2>&1
sh scripts/rome/zsre_llama3_100_1_seed45.sh > log/rome_zsre_llama3_100_1_seed45.log 2>&1
sh scripts/rome/zsre_gpt2xl_100_1_seed45.sh > log/rome_zsre_gpt2xl_100_1_seed45.log 2>&1
sh scripts/rome/zsre_gptj_100_1_seed45.sh > log/rome_zsre_gptj_100_1_seed45.log 2>&1
sh scripts/rome/zsre_llama2_100_1_seed45.sh > log/rome_zsre_llama2_100_1_seed45.log 2>&1

sh scripts/rome/cf_mistral_100_50_seed45.sh > log/rome_cf_mistral_100_50_seed45.log 2>&1
sh scripts/rome/cf_llama3_100_50_seed45.sh > log/rome_cf_llama3_100_50_seed45.log 2>&1
sh scripts/rome/cf_gpt2xl_100_50_seed45.sh > log/rome_cf_gpt2xl_100_50_seed45.log 2>&1
sh scripts/rome/cf_gptj_100_50_seed45.sh > log/rome_cf_gptj_100_50_seed45.log 2>&1
sh scripts/rome/cf_llama2_100_50_seed45.sh > log/rome_cf_llama2_100_50_seed45.log 2>&1
sh scripts/rome/zsre_mistral_100_50_seed45.sh > log/rome_zsre_mistral_100_50_seed45.log 2>&1
sh scripts/rome/zsre_llama3_100_50_seed45.sh > log/rome_zsre_llama3_100_50_seed45.log 2>&1
sh scripts/rome/zsre_gpt2xl_100_50_seed45.sh > log/rome_zsre_gpt2xl_100_50_seed45.log 2>&1
sh scripts/rome/zsre_gptj_100_50_seed45.sh > log/rome_zsre_gptj_100_50_seed45.log 2>&1
sh scripts/rome/zsre_llama2_100_50_seed45.sh > log/rome_zsre_llama2_100_50_seed45.log 2>&1