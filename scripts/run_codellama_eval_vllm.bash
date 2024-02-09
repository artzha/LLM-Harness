#!/bin/bash
python human_eval/evaluate.py model=codellama data.save_completion_path=model_outputs/humaneval/codellama_vllm.jsonl
