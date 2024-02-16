import hydra
from omegaconf import DictConfig, OmegaConf


from transformers import AutoTokenizer, TrainingArguments
import time
import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import datasets

@hydra.main(version_base=None, config_path="configs", config_name="llm")
def main(cfg: DictConfig):
    """
    Uses trl to finetune a pretrained model on the multi plt dataset using the SFTTrainer
    """
    print(OmegaConf.to_yaml(cfg))

    # Load the dataset
    dataset = datasets.load_from_disk(cfg.data.root_path)

    training_args = TrainingArguments(
        **cfg.trainer.training_args.kwargs
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, use_fast=False, max_length=cfg.model.training_params.kwargs.max_seq_length)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation = True

    # Create the data collator
    # data_collator = DataCollatorForCompletionOnlyLM(
    #     response_template=cfg.data.response_template,
    #     tokenizer=tokenizer)

    # Load the model
    trainer = SFTTrainer(
        cfg.model.name,
        tokenizer=tokenizer,
        # data_collator=data_collator,
        train_dataset=dataset["train"],
        args=training_args,
        **cfg.model.training_params.kwargs
    )

    trainer.train()
    trainer.save_model(cfg.trainer.ckpt_path)


if __name__ == "__main__":
    main()