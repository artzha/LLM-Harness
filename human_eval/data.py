import os
from os.path import join
import gzip
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Torch deps
import torch
from torch.utils.data import Dataset, DataLoader

import human_eval.data_utils as du

class HumanevalDataset(Dataset):
    def __init__(self, root_path, max_length=512):
        self.root_path = root_path
        self.max_length = max_length

        self.HUMANEVAL_PATH = join(root_path, "HumanEval.jsonl.gz") 

    def __len__(self):
        return len(self.data)
    
    def load_dataset(self):
        """
        Return a list of dictionaries, each dictionary contains the following keys:
            'task_id': (str)Unique identifier for the task
            'prompt': (str) Input prompt to generator (in code format)
            'entry_point' (str)
            'canonical_solution' (str) The correct code solution to the prompt
            'test' (str) The test case to evaluate the solution. Contains METADATA dict
        """
        assert os.path.exists(self.HUMANEVAL_PATH ), f"File {self.HUMANEVAL_PATH} not found."

        self.data = [x for _, x in enumerate(du.stream_jsonl(self.HUMANEVAL_PATH))]

    def collate_fn(self, batch):
        
        task_ids = [x['task_id'] for x in batch]
        prompts = [x['prompt'] for x in batch]
        entry_points = [x['entry_point'] for x in batch]
        canonical_solutions = [x['canonical_solution'] for x in batch]
        tests = [x['test'] for x in batch]

        return {
            "task_id": task_ids,
            "prompt": prompts,
            "entry_point": entry_points,
            "canonical_solution": canonical_solutions,
            "test": tests
        }
    
    def get_dataloader(self, shuffle=False, batch_size=8):
        return DataLoader(self, shuffle=shuffle, batch_size=batch_size, collate_fn=self.collate_fn)

    def __getitem__(self, idx):
        """
        TODO: In the future we may want to return the tokenized version of the prompt and the test case
        """
        return {
            "task_id": self.data[idx]['task_id'],
            "prompt": self.data[idx]['prompt'].strip(),
            "entry_point": self.data[idx]['entry_point'],
            "canonical_solution": self.data[idx]['canonical_solution'],
            "test": self.data[idx]['test']
        }
        
    
if __name__ == '__main__':
    dataset = HumanevalDataset(root_path="data")
    dataset.load_dataset()

    dataloader = dataset.get_dataloader(batch_size=6)

    for batch_idx, batch in enumerate(dataloader):
        print("Prompts size", len(batch['prompt']))
        print("Prompts ", batch['prompt'])

        if batch_idx > 5:
            break