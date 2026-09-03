#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

sh scripts/alphaedit/cf_gpt2xl_100_1_seed45.sh > log/alphaedit_cf_gpt2xl_100_1_seed45.log 2>&1
sh scripts/alphaedit/cf_gptj_100_1_seed45.sh > log/alphaedit_cf_gptj_100_1_seed45.log 2>&1
sh scripts/alphaedit/cf_llama2_100_1_seed45.sh > log/alphaedit_cf_llama2_100_1_seed45.log 2>&1
sh scripts/alphaedit/cf_mistral_100_1_seed45.sh > log/alphaedit_cf_mistral_100_1_seed45.log 2>&1
sh scripts/alphaedit/cf_llama3_100_1_seed45.sh > log/alphaedit_cf_llama3_100_1_seed45.log 2>&1
sh scripts/alphaedit/zsre_gpt2xl_100_1_seed45.sh > log/alphaedit_zsre_gpt2xl_100_1_seed45.log 2>&1
sh scripts/alphaedit/zsre_gptj_100_1_seed45.sh > log/alphaedit_zsre_gptj_100_1_seed45.log 2>&1
sh scripts/alphaedit/zsre_llama2_100_1_seed45.sh > log/alphaedit_zsre_llama2_100_1_seed45.log 2>&1
sh scripts/alphaedit/zsre_mistral_100_1_seed45.sh > log/alphaedit_zsre_mistral_100_1_seed45.log 2>&1
sh scripts/alphaedit/zsre_llama3_100_1_seed45.sh > log/alphaedit_zsre_llama3_100_1_seed45.log 2>&1