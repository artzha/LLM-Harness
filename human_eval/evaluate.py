import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# configuration imports
import hydra
from omegaconf import DictConfig, OmegaConf

import data_utils as du
import eval_utils as eu
from data import HumanevalDataset

# if __name__ == "__main__":


#     res = eu.evaluate_functional_correctness(
#         sample_file=args.sample_file,
#         k=[1],
#         problem_file="data/HumanEval.jsonl",
#         timeout=3)
#     print(res)

@hydra.main(version_base=None, config_path="../configs", config_name="llm")
def generate(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load the predictions
    responses = [ response for response in du.stream_jsonl(cfg.data.save_completion_path)]

    # Load the dataset
    dataset = HumanevalDataset(root_path=cfg.data.root_path)
    dataset.load_dataset()

    # Sanity check evaluator
    # k = [1]
    # stats = eu.evaluate_functional_correctness(
    #     sample_file='data/example_samples.jsonl',
    #     problems='data/example_problem.jsonl',
    #     k=k,
    #     n_workers=4,
    #     timeout=3.0
    # )

    # Evaluate the samples
    k = [1]
    stats = eu.evaluate_functional_correctness(
        sample_file=cfg.data.save_completion_path,
        problems=dataset,
        k=k,
        n_workers=4,
        timeout=3.0
    )

    for metric, score in stats.items():
        print(f'Model {cfg.model.name} achieved a {metric} score of {score}')


if __name__ == '__main__':
    generate()