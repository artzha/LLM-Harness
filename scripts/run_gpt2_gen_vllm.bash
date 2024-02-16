#!/bin/bash
python human_eval/generate.py model=gpt2 data=multipl-t-python data.save_completion_path=model_outputs/multipl-t-python/gpt2_vllm.jsonl
