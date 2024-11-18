# coding=utf-8

import logging
import math
import os
import random
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import logging
from typing import  Dict, Any
import yaml

from datasets import DatasetDict
from transformers import (
    AdamW,
    DataCollatorWithPadding,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from accelerate import Accelerator
# Append the path for custom modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import custom modules
from cfinetuning import (
    CTrainingArguments,
    DataConfig,
    DatasetConfig,
    DatasetProcessor,
    ModelConfig,
    ModelLoader,
    PromptTemplate,
    train,
    evaluate,
    Split
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(file_path: Path) -> Dict[str, Any]:
     """Load configuration from a YAML file."""
     try:
         with open(file_path, 'r', encoding='utf-8') as file:
             return yaml.safe_load(file)
     except (yaml.YAMLError, FileNotFoundError) as e:
         logger.error(f"Error loading config: {e}")
         raise

def finetune():
    """Fine-tuning a pre-trained model on a downstream task.

    Args:
      accelerator: An instance of an accelerator for distributed training (on
        multi-GPU, TPU) or mixed precision training.
    """
    set_seed(seed=0)
    accelerator = Accelerator()
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    # Constants
    CONFIG_FILE_PATH = Path("E:/LLMS/Fine-tuning/LlmsComponents/Fine_tuning/ccongfig.yml")
    config_data = load_config(CONFIG_FILE_PATH)
    
    # Extract configuration arguments using nested get method for better readability
    data_loader_args = config_data.get('config1', {}).get('datasetconfig', {}).get('DataConfig', {})
    data_preprocess_args = config_data.get('config1', {}).get('datasetconfig', {}).get('DatasetConfig', {})
    prompt_template_args = config_data.get('config1', {}).get('promptconfig', {}).get('PromptTemplate', {})
    model_loader_args = config_data.get('config1', {}).get('modelconfig', {}).get('ModelConfig', {})
    ctrainings_args = config_data.get('config1', {}).get('trainingconfig', {}).get('CTrainingArguments', {})
    
    # Initialize configuration objects
    data_config = DataConfig(**data_loader_args)  # type: ignore
    dataset_config = DatasetConfig(**data_preprocess_args)  # type: ignore
    prompt_template = PromptTemplate(**prompt_template_args)  # type: ignore
    model_config = ModelConfig(**model_loader_args)  # type: ignore
    trainging_config=CTrainingArguments(**ctrainings_args) # type: ignore
    
    # Initialize and load model and tokenizer
    model_loader = ModelLoader(model_config)
    model, tokenizer = model_loader.load_model_and_tokenizer()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = model_loader.get_config()
    
    # Initialize dataset processor with configurations
    dataset_processor = DatasetProcessor(
        dataloader_config=data_config,
        dataset_config=dataset_config,
        prompt_template=prompt_template,
        tokenizer=tokenizer
    )
    
    # Process datasets
    processed_datasets: DatasetDict = dataset_processor.process_dataset()

    # Setup logging, we only want one process per machine to log things on the
    # screen. accelerator.is_local_main_process is only True for one process per
    # machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

  
    args=trainging_config
    train_dataset=processed_datasets['train']
    test_dataset=processed_datasets['test']
    eval_dataset=processed_datasets['test']
    infer_dataset=processed_datasets['test']
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info("Sample %d of the training set: %s.", index, train_dataset[index])

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data
        # collator that will just convert everything to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by
        # padding to the maximum length of the samples passed). When using mixed
        # precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple of
        # 8s, which will enable the use of Tensor Cores on NVIDIA hardware with
        # compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    eval_dataloader, test_dataloader, infer_dataloader = None, None, None

    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator
        )

    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator
        )

    if infer_dataset is not None:
        infer_dataloader = DataLoader(
            infer_dataset, batch_size=args.per_device_eval_batch_size, collate_fn=data_collator
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, infer_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, infer_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab its
    # length below (cause its length will be shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_steps == -1:
        args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Train
    completed_steps, avg_train_loss = train(
        args, accelerator, model, tokenizer, train_dataloader, optimizer, lr_scheduler, eval_dataloader
    )
    accelerator.wait_for_everyone()
    logger.info("Training job completed: completed_steps = %d, avg_train_loss = %f", completed_steps, avg_train_loss)

    args.model_name_or_path = os.path.join(args.output_dir, "best-checkpoint")
    logger.info("Loading the best checkpoint: %s", args.model_name_or_path)
    model = accelerator.prepare(model)

    if args.do_eval:
        # Evaluate
        if eval_dataloader is not None:
            logger.info("***** Running evaluation on the eval data using the best checkpoint *****")
            eval_results = evaluate(args, accelerator, eval_dataloader, Split.EVAL.value, model, "best-checkpoint")
            avg_eval_loss = eval_results["avg_eval_loss"]
            eval_metric = eval_results[args.eval_metric]
            logger.info("Evaluation job completed: avg_eval_loss = %f", avg_eval_loss)
            logger.info("Evaluation result for the best checkpoint: %s = %f", args.eval_metric, eval_metric)

        if test_dataloader is not None:
            logger.info("***** Running evaluation on the test data using the best checkpoint *****")
            eval_results = evaluate(args, accelerator, test_dataloader, Split.TEST.value, model, "best-checkpoint")
            avg_eval_loss = eval_results["avg_eval_loss"]
            eval_metric = eval_results[args.eval_metric]
            logger.info("Test job completed: avg_test_loss = %f", avg_eval_loss)
            logger.info("Test result for the best checkpoint: %s = %f", args.eval_metric, eval_metric)

    if args.do_predict:
        # Predict
        if infer_dataloader is not None:
            logger.info("***** Running inference using the best checkpoint *****")
            evaluate(
                args, accelerator, infer_dataloader, Split.INFER.value, model, "best-checkpoint", has_labels=False
            )
            logger.info("Inference job completed.")

    # Release all references to the internal objects stored and call the garbage
    # collector. You should call this method between two trainings with different
    # models/optimizers.
    accelerator.free_memory()

finetune()