import os
import time

# configuration imports
import hydra
from omegaconf import DictConfig, OmegaConf

# LLM Imports
from vllm import LLM, SamplingParams

# Custom Imports
from data import HumanevalDataset
import data_utils as du
from tqdm import tqdm

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]
stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]

class LLM_Wrapper(object):
    def __init__(self, cfg: DictConfig):
        self.model = LLM(
            model=cfg.model.name
        )
        self.model_name = cfg.model.name
        self.api_name = cfg.model.api_name

        # Sampling Params
        self.sampling_params = SamplingParams(**cfg.model.sampling_params.kwargs)

    def __call__(self, prompt):
        # TODO: Modify this for multiple prompts
        outputs     = self.model.generate(prompt[0], self.sampling_params)

        completions   =  [_truncate_code_at_stopwords(output.outputs[0].text, stop_words) for output in outputs]

        return completions

@hydra.main(version_base=None, config_path="../configs", config_name="llm")
def generate(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load the model
    model = LLM_Wrapper(cfg)

    # Load the dataset
    dataset = HumanevalDataset(root_path=cfg.data.root_path)
    dataset.load_dataset()

    # Generate the samples
    data_loader = dataset.get_dataloader(shuffle=cfg.trainer.shuffle, batch_size=cfg.trainer.batch_size)

    start_time = time.time()
    print("Time to generate completions")
    saved_data = []
    for batch in tqdm(data_loader):
        task_id, prompts = batch['task_id'], batch['prompt'] 
        completions = model(prompts)

        # Save responses to file
        saved_data.extend([
            {
                "task_id": task_id[i],
                "completion": completions[i]
            } for i in range(len(task_id))
        
        ])
        print(prompts[0])
        print(completions[0])
    print("Time to generate completions: ", time.time() - start_time)
    # Save the responses
    completion_path = cfg.data.save_completion_path
    save_dir = os.path.dirname(completion_path)
    os.makedirs(save_dir, exist_ok=True)
    du.write_jsonl(completion_path, saved_data, append=False)

if __name__ == '__main__':
    generate()