#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

sh scripts/rome-simie/cf_mistral_100_1_seed45.sh > log/rome-simie_cf_mistral_100_1_seed45.log 2>&1
sh scripts/rome-simie/cf_llama3_100_1_seed45.sh > log/rome-simie_cf_llama3_100_1_seed45.log 2>&1
sh scripts/rome-simie/cf_gpt2xl_100_1_seed45.sh > log/rome-simie_cf_gpt2xl_100_1_seed45.log 2>&1
sh scripts/rome-simie/cf_gptj_100_1_seed45.sh > log/rome-simie_cf_gptj_100_1_seed45.log 2>&1
sh scripts/rome-simie/cf_llama2_100_1_seed45.sh > log/rome-simie_cf_llama2_100_1_seed45.log 2>&1
sh scripts/rome-simie/zsre_mistral_100_1_seed45.sh > log/rome-simie_zsre_mistral_100_1_seed45.log 2>&1
sh scripts/rome-simie/zsre_llama3_100_1_seed45.sh > log/rome-simie_zsre_llama3_100_1_seed45.log 2>&1
sh scripts/rome-simie/zsre_gpt2xl_100_1_seed45.sh > log/rome-simie_zsre_gpt2xl_100_1_seed45.log 2>&1
sh scripts/rome-simie/zsre_gptj_100_1_seed45.sh > log/rome-simie_zsre_gptj_100_1_seed45.log 2>&1
sh scripts/rome-simie/zsre_llama2_100_1_seed45.sh > log/rome-simie_zsre_llama2_100_1_seed45.log 2>&1

sh scripts/rome-simie/cf_mistral_100_50_seed45.sh > log/rome-simie_cf_mistral_100_50_seed45.log 2>&1
sh scripts/rome-simie/cf_llama3_100_50_seed45.sh > log/rome-simie_cf_llama3_100_50_seed45.log 2>&1
sh scripts/rome-simie/cf_gpt2xl_100_50_seed45.sh > log/rome-simie_cf_gpt2xl_100_50_seed45.log 2>&1
sh scripts/rome-simie/cf_gptj_100_50_seed45.sh > log/rome-simie_cf_gptj_100_50_seed45.log 2>&1
sh scripts/rome-simie/cf_llama2_100_50_seed45.sh > log/rome-simie_cf_llama2_100_50_seed45.log 2>&1
sh scripts/rome-simie/zsre_mistral_100_50_seed45.sh > log/rome-simie_zsre_mistral_100_50_seed45.log 2>&1
sh scripts/rome-simie/zsre_llama3_100_50_seed45.sh > log/rome-simie_zsre_llama3_100_50_seed45.log 2>&1
sh scripts/rome-simie/zsre_gpt2xl_100_50_seed45.sh > log/rome-simie_zsre_gpt2xl_100_50_seed45.log 2>&1
sh scripts/rome-simie/zsre_gptj_100_50_seed45.sh > log/rome-simie_zsre_gptj_100_50_seed45.log 2>&1
sh scripts/rome-simie/zsre_llama2_100_50_seed45.sh > log/rome-simie_zsre_llama2_100_50_seed45.log 2>&1