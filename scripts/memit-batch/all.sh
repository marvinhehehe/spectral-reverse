#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

sh scripts/memit-batch/cf_mistral_100_1_seed45.sh > log/memit-batch_cf_mistral_100_1_seed45.log 2>&1
sh scripts/memit-batch/cf_llama3_100_1_seed45.sh > log/memit-batch_cf_llama3_100_1_seed45.log 2>&1
sh scripts/memit-batch/cf_gpt2xl_100_1_seed45.sh > log/memit-batch_cf_gpt2xl_100_1_seed45.log 2>&1
sh scripts/memit-batch/cf_gptj_100_1_seed45.sh > log/memit-batch_cf_gptj_100_1_seed45.log 2>&1
sh scripts/memit-batch/cf_llama2_100_1_seed45.sh > log/memit-batch_cf_llama2_100_1_seed45.log 2>&1
sh scripts/memit-batch/zsre_mistral_100_1_seed45.sh > log/memit-batch_zsre_mistral_100_1_seed45.log 2>&1
sh scripts/memit-batch/zsre_llama3_100_1_seed45.sh > log/memit-batch_zsre_llama3_100_1_seed45.log 2>&1
sh scripts/memit-batch/zsre_gpt2xl_100_1_seed45.sh > log/memit-batch_zsre_gpt2xl_100_1_seed45.log 2>&1
sh scripts/memit-batch/zsre_gptj_100_1_seed45.sh > log/memit-batch_zsre_gptj_100_1_seed45.log 2>&1
sh scripts/memit-batch/zsre_llama2_100_1_seed45.sh > log/memit-batch_zsre_llama2_100_1_seed45.log 2>&1

sh scripts/memit-batch/cf_mistral_100_50_seed45.sh > log/memit-batch_cf_mistral_100_50_seed45.log 2>&1
sh scripts/memit-batch/cf_llama3_100_50_seed45.sh > log/memit-batch_cf_llama3_100_50_seed45.log 2>&1
sh scripts/memit-batch/cf_gpt2xl_100_50_seed45.sh > log/memit-batch_cf_gpt2xl_100_50_seed45.log 2>&1
sh scripts/memit-batch/cf_gptj_100_50_seed45.sh > log/memit-batch_cf_gptj_100_50_seed45.log 2>&1
sh scripts/memit-batch/cf_llama2_100_50_seed45.sh > log/memit-batch_cf_llama2_100_50_seed45.log 2>&1
sh scripts/memit-batch/zsre_mistral_100_50_seed45.sh > log/memit-batch_zsre_mistral_100_50_seed45.log 2>&1
sh scripts/memit-batch/zsre_llama3_100_50_seed45.sh > log/memit-batch_zsre_llama3_100_50_seed45.log 2>&1
sh scripts/memit-batch/zsre_gpt2xl_100_50_seed45.sh > log/memit-batch_zsre_gpt2xl_100_50_seed45.log 2>&1
sh scripts/memit-batch/zsre_gptj_100_50_seed45.sh > log/memit-batch_zsre_gptj_100_50_seed45.log 2>&1
sh scripts/memit-batch/zsre_llama2_100_50_seed45.sh > log/memit-batch_zsre_llama2_100_50_seed45.log 2>&1