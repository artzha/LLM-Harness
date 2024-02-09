import os
import time

# configuration imports
import hydra
from omegaconf import DictConfig, OmegaConf

# LLM Imports
from vllm import LLM, SamplingParams

# class LLM_Wrapper(object):
#     def __init__(self, cfg: DictConfig):
#         self.model = LLM(
#             model=cfg.model.name
#         )
#         self.model_name = cfg.model.name
#         self.api_name = cfg.model.api_name

#         # Sampling Params
#         self.sampling_params = SamplingParams(**cfg.model.sampling_params.kwargs)

#     def __call__(self, prompt):
#         outputs     = self.model.generate(prompt[0], self.sampling_params)

#         completions   =  [_truncate_code_at_stopwords(output.outputs[0].text, stop_words) for output in outputs]

#         return completions

@hydra.main(version_base=None, config_path="../configs", config_name="llm")
def generate(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


    # Load the model
    model = LLM(cfg.model.name)
    sampling_params = SamplingParams(**cfg.model.sampling_params.kwargs)

    print("Initializing chat server with model: ", cfg.model.name)
 
    # Start infinite loop to listen for user prompts, process them, then print completions
    while True:
        prompt = input("User: ")
        completions = model.generate(prompt, sampling_params)
        print("Assistant: ", completions[0].outputs[0].text)

    # for batch in tqdm(data_loader):
    #     task_id, prompts = batch['task_id'], batch['prompt'] 
    #     completions = model(prompts)

    #     # Save responses to file
    #     saved_data.extend([
    #         {
    #             "task_id": task_id[i],
    #             "completion": completions[i]
    #         } for i in range(len(task_id))
        
    #     ])
    #     print(prompts[0])
    #     print(completions[0])
    # print("Time to generate completions: ", time.time() - start_time)
    # # Save the responses
    # completion_path = cfg.data.save_completion_path
    # save_dir = os.path.dirname(completion_path)
    # os.makedirs(save_dir, exist_ok=True)
    # du.write_jsonl(completion_path, saved_data, append=False)

if __name__ == '__main__':
    generate()