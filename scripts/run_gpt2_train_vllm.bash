#!/bin/bash
python train.py model=gpt2 data=multipl-t-python data.save_completion_path=model_outputs/multipl-t-python/gpt2_vllm.jsonl trainer=gpt2_standard
