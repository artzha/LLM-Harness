#!/bin/bash
python human_eval/generate.py model=llamav2 data.save_completion_path=model_outputs/humaneval/llamav2_vllm.jsonl
