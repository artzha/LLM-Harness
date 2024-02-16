#!/bin/bash
python human_eval/evaluate.py model=gpt2 data=humaneval data.save_completion_path=model_outputs/multipl-t-python/gpt2_vllm.jsonl 