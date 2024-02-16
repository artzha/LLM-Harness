import hydra
from omegaconf import DictConfig, OmegaConf


from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import argparse
from trl import SFTTrainer
import datasets

def formatting_func(example):
    prompt = "### Human:\n" + example["prompt"].strip() + "\n### Assistant: " 
    return prompt

@hydra.main(version_base=None, config_path="../configs", config_name="llm")
def main(cfg: DictConfig):
    """
    Uses trl to finetune a pretrained model on the multi plt dataset using the SFTTrainer
    """
    print(OmegaConf.to_yaml(cfg))

    # Load the dataset
    dataset = datasets.load_from_disk(cfg.data.root_path)
    import pdb; pdb.set_trace()

    # Load the model
    trainer = SFTTrainer(
        cfg.model.name,
        train_dataset=dataset,
        **cfg.model.training_params.kwargs
    )
    trainer.train()    


if __name__ == "__main__":
    main()