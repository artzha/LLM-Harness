from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from human_eval.data_utils import write_jsonl
import time
import argparse

def formatting_func(example):
    prompt = "### Human:\n" + example["prompt"].strip() + "\n### Assistant: " 
    return prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", type=str, default="openai-community/gpt2")
    parser.add_argument("-s", "--save_name", type=str, default="tmp.jsonl")
    args = parser.parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    humaneval = load_dataset("openai_humaneval")["test"]

    def _truncate_code_at_stopwords(code, stop_words):
        min_stop_idx = len(code)
        for stop_word in stop_words:
            stop_index = code.find(stop_word)
            if 0 <= stop_index < min_stop_idx:
                min_stop_idx = stop_index
        return code[:min_stop_idx]
    stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]

    start_time = time.time()
    print("Time to generate completions")
    saved_data = []
    for idx, example in enumerate(humaneval):
        print(f"generating: {idx}")
        task_id = example["task_id"]
        prompt = example["prompt"].strip() # formatting_func(example) # adding formatting func will cause more syntax errors
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids, 
            max_length=512, 
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
            )
        generation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = _truncate_code_at_stopwords(generation[len(prompt):], stop_words)
        print(prompt)
        print(completion)

        saved_data.append({
            "task_id": task_id,
            "completion": completion,
        })
    print("Time to generate completions: ", time.time() - start_time)
    write_jsonl(args.save_name, saved_data, append=False)