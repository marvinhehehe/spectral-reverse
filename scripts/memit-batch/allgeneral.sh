#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python glue_eval/cola_eval.py > log/llama3_memit_batch_cola_eval.log 2>&1
python glue_eval/mmlu_eval.py > log/llama3_memit_batch_mmlu_eval.log 2>&1
python glue_eval/mrpc_eval.py > log/llama3_memit_batch_mrpc_eval.log 2>&1
python glue_eval/nli_eval.py > log/llama3_memit_batch_nli_eval.log 2>&1
python glue_eval/rte_eval.py > log/llama3_memit_batch_rte_eval.log 2>&1
python glue_eval/sst_eval.py > log/llama3_memit_batch_sst_eval.log 2>&1