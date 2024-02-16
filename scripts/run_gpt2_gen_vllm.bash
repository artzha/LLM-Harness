#!/bin/bash
python human_eval/generate.py model=gpt2 model.weights_path=model_outputs/gpt2/training/checkpoint-2900 data=humaneval data.save_completion_path=model_outputs/multipl-t-python/gpt2_vllm.jsonl trainer=gpt2_standard
