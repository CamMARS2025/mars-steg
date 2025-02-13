from __future__ import annotations

# import wandb
import sys
import torch

from transformers import set_seed

from trl import PPOConfig, PPOTrainer

from mars_steg.config import ConfigLoader, ExperimentArgs
from mars_steg.utils.common import (
    get_dataloaders,
    get_tokenizer,
    get_model,
    get_device,
    evaluate_cot_gap_summary,
)


experiment_args_path = sys.argv[1]
config_path = sys.argv[2]   # e.g. "mars_steg/configs/config.yaml"


device = get_device()

with torch.no_grad():

    model_config, optimizer_config, train_config, generation_config = ConfigLoader.load_config(config_path)

    generation_config.batch_size = train_config.batch_size

    # PPO Configuration
    config = PPOConfig(
        model_name=model_config.model_name,
        learning_rate=optimizer_config.learning_rate,
        log_with=train_config.log_with,
        ppo_epochs=train_config.ppo_epochs,
        mini_batch_size=train_config.mini_batch_size,
        batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
    )


    experiment_args = ExperimentArgs.from_yaml(experiment_args_path)


    # wandb.init(
    #     project="MARS-STEGO",
    #     config={
    #         'ppo_config': config,
    #         'model_config': model_config,
    #         'optimizer_config': optimizer_config,
    #         'train_config': train_config,
    #         'experiment_args': experiment_args,
    #     }
    # )

    # Set random seed for reproducibility
    set_seed(config.seed)


    tokenizer = get_tokenizer(model_config.model_name)
    generation_config.pad_token_id = tokenizer.eos_token_id 
    model = get_model(model_config, tokenizer)

    generation_kwargs = generation_config.to_dict()

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders(
        dataset_class_name = experiment_args.dataset_class_name,
        penalisation_class_name = experiment_args.penalisation_class_name,
        dataset_class_kwargs = experiment_args.dataset_class_kwargs,
        penalisation_class_kwargs = experiment_args.penalisation_class_kwargs,
        train_proportion = 1,
        validation_proportion = 0,
        batch_size = train_config.batch_size,
        reference_model = None, # 
    )


    ppo_trainer = PPOTrainer(
        config,
        model.model,
        ref_model=None,
        tokenizer=tokenizer,
    )


    cot_gap = evaluate_cot_gap_summary(
        ppo_trainer = ppo_trainer, 
        tokenizer = tokenizer, 
        dataloader = train_loader, 
        system_prompt = val_dataset.system_prompt,
        model_class = model, 
        device = device, 
        generation_kwargs = generation_kwargs,
        output_delimitation_tokens= output_delimitation_tokens,
    )


    # TODO: better logging here!!
    print(cot_gap.summary)

