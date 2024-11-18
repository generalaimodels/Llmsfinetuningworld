# coding=utf-8

import json
import logging
import os
import shutil
import numpy as np
import pandas as pd
import torch
from typing import Dict,Optional,Tuple
import os
import json
import torch
import numpy as np
import pandas as pd
from evaluate import load
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_metric
from tqdm.auto import tqdm
from transformers.file_utils import ExplicitEnum
from transformers.trainer_utils import IntervalStrategy


logger = logging.getLogger(__name__)


class Split(ExplicitEnum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"
    INFER = "infer"



def train(args, accelerator, model, tokenizer, train_dataloader, optimizer, lr_scheduler, eval_dataloader=None):
    """Train a model on the given training data."""

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", args.num_examples[Split.TRAIN.value])
    logger.info("  Instantaneous batch size per device = %d", args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_local_main_process)

    checkpoints = None
    eval_results = None
    best_checkpoint = None
    best_eval_result = None
    early_stopping_patience_counter = 0
    should_training_stop = False
    epoch = 0
    completed_steps = 0
    train_loss = 0.0
    model.zero_grad()

    for _ in range(args.num_train_epochs):
        epoch += 1
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(input_ids=batch['input_ids'],labels=batch["labels"])
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss += loss.item()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                # Evaluate during training
                if (
                    eval_dataloader is not None
                    and args.eval_strategy == IntervalStrategy.STEPS.value
                    and args.eval_steps > 0
                    and completed_steps % args.eval_steps == 0
                ):
                    accelerator.wait_for_everyone()
                    new_checkpoint = f"checkpoint-{IntervalStrategy.STEPS.value}-{completed_steps}"
                    new_eval_result = evaluate(args, accelerator, eval_dataloader, "eval", model, new_checkpoint)[
                        args.eval_metric
                    ]
                    logger.info(
                        "Evaluation result at step %d: %s = %f", completed_steps, args.eval_metric, new_eval_result
                    )
                    if checkpoints is None:
                        checkpoints = np.array([new_checkpoint])
                        eval_results = np.array([new_eval_result])
                        best_checkpoint = new_checkpoint
                        best_eval_result = new_eval_result
                    else:
                        if new_eval_result - best_eval_result > args.early_stopping_threshold:
                            best_checkpoint = new_checkpoint
                            best_eval_result = new_eval_result
                            early_stopping_patience_counter = 0
                        else:
                            if new_eval_result == best_eval_result:
                                best_checkpoint = new_checkpoint
                                best_eval_result = new_eval_result
                            early_stopping_patience_counter += 1

                        if early_stopping_patience_counter >= args.early_stopping_patience:
                            should_training_stop = True

                        checkpoints = np.append(checkpoints, [new_checkpoint], axis=0)
                        eval_results = np.append(eval_results, [new_eval_result], axis=0)
                        sorted_ids = np.argsort(eval_results)
                        eval_results = eval_results[sorted_ids]
                        checkpoints = checkpoints[sorted_ids]

                    if len(checkpoints) > args.keep_checkpoint_max:
                        # Delete the current worst checkpoint
                        checkpoint_to_remove, *checkpoints = checkpoints
                        eval_results = eval_results[1:]
                        if checkpoint_to_remove != new_checkpoint:
                            if accelerator.is_main_process:
                                shutil.rmtree(os.path.join(args.output_dir, checkpoint_to_remove), ignore_errors=True)
                            accelerator.wait_for_everyone()

                    if new_checkpoint in checkpoints:
                        # Save model checkpoint
                        checkpoint_output_dir = os.path.join(args.output_dir, new_checkpoint)
                        if accelerator.is_main_process:
                            if not os.path.exists(checkpoint_output_dir):
                                os.makedirs(checkpoint_output_dir)
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=accelerator.save)
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(checkpoint_output_dir)
                            logger.info("Saving model checkpoint to %s", checkpoint_output_dir)

            if completed_steps >= args.max_steps:
                break

            if should_training_stop:
                break

        # Evaluate during training
        if eval_dataloader is not None and args.eval_strategy == IntervalStrategy.EPOCH.value:
            accelerator.wait_for_everyone()
            new_checkpoint = f"checkpoint-{IntervalStrategy.EPOCH.value}-{epoch}"
            new_eval_result = evaluate(args, accelerator, eval_dataloader, "eval", model, new_checkpoint)[
                args.eval_metric
            ]
            logger.info("Evaluation result at epoch %d: %s = %f", epoch, args.eval_metric, new_eval_result)

            if checkpoints is None:
                checkpoints = np.array([new_checkpoint])
                eval_results = np.array([new_eval_result])
                best_checkpoint = new_checkpoint
                best_eval_result = new_eval_result
            else:
                if new_eval_result - best_eval_result > args.early_stopping_threshold:
                    best_checkpoint = new_checkpoint
                    best_eval_result = new_eval_result
                    early_stopping_patience_counter = 0
                else:
                    if new_eval_result == best_eval_result:
                        best_checkpoint = new_checkpoint
                        best_eval_result = new_eval_result
                    early_stopping_patience_counter += 1

                if early_stopping_patience_counter >= args.early_stopping_patience:
                    should_training_stop = True

                checkpoints = np.append(checkpoints, [new_checkpoint], axis=0)
                eval_results = np.append(eval_results, [new_eval_result], axis=0)
                sorted_ids = np.argsort(eval_results)
                eval_results = eval_results[sorted_ids]
                checkpoints = checkpoints[sorted_ids]

            if len(checkpoints) > args.keep_checkpoint_max:
                # Delete the current worst checkpoint
                checkpoint_to_remove, *checkpoints = checkpoints
                eval_results = eval_results[1:]
                if checkpoint_to_remove != new_checkpoint:
                    if accelerator.is_main_process:
                        shutil.rmtree(os.path.join(args.output_dir, checkpoint_to_remove), ignore_errors=True)
                    accelerator.wait_for_everyone()

            if new_checkpoint in checkpoints:
                # Save model checkpoint
                checkpoint_output_dir = os.path.join(args.output_dir, new_checkpoint)
                if accelerator.is_main_process:
                    if not os.path.exists(checkpoint_output_dir):
                        os.makedirs(checkpoint_output_dir)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(checkpoint_output_dir)
                    logger.info("Saving model checkpoint to %s", checkpoint_output_dir)

        if completed_steps >= args.max_steps:
            break

        if should_training_stop:
            break

    if best_checkpoint is not None:
        # Save the best checkpoint
        logger.info("Best checkpoint: %s", best_checkpoint)
        logger.info("Best evaluation result: %s = %f", args.eval_metric, best_eval_result)
        best_checkpoint_output_dir = os.path.join(args.output_dir, best_checkpoint)
        if accelerator.is_main_process:
            shutil.move(best_checkpoint_output_dir, os.path.join(args.output_dir, "best-checkpoint"))
            shutil.rmtree(best_checkpoint_output_dir, ignore_errors=True)
        accelerator.wait_for_everyone()

    else:
        # Assume that the last checkpoint is the best checkpoint and save it
        checkpoint_output_dir = os.path.join(args.output_dir, "best-checkpoint")
        if not os.path.exists(checkpoint_output_dir):
            os.makedirs(checkpoint_output_dir)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(checkpoint_output_dir)
            logger.info("Saving model checkpoint to %s", checkpoint_output_dir)
    return completed_steps, train_loss / completed_steps



def compute_metrics(metric, eval_pred: Tuple) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Flatten predictions and labels, ignoring padded tokens (`-100`) in the labels
    flattened_predictions = []
    flattened_labels = []

    for pred, label in zip(predictions, labels):
        valid_indices = label != -100
        valid_preds = pred[valid_indices]
        valid_labels = label[valid_indices]

        flattened_predictions.extend(valid_preds.tolist())
        flattened_labels.extend(valid_labels.tolist())

    return metric.compute(predictions=flattened_predictions, references=flattened_labels)

def evaluate(
    args,
    accelerator: Accelerator,
    dataloader: DataLoader,
    eval_set: str,
    model: torch.nn.Module,
    checkpoint: str,
    has_labels: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model checkpoint on the given evaluation data.

    Args:
        args: Arguments containing configuration settings.
        accelerator: Accelerator object for distributed training.
        dataloader: DataLoader for the evaluation dataset.
        eval_set: Name of the evaluation set.
        model: The model to evaluate.
        checkpoint: Checkpoint identifier.
        has_labels: Whether the dataset has labels.

    Returns:
        A dictionary containing evaluation results.
    """
    eval_metric = load(args.eval_metric)
    completed_steps = 0
    eval_loss = 0.0
    all_logits = []
    all_labels = []

    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], labels=batch["labels"])

        eval_loss += outputs.loss.item()
        logits = outputs.logits
        logits = accelerator.gather(logits).detach().cpu().numpy()
        all_logits.append(logits)

        if has_labels:
            labels = batch["labels"]
            labels = accelerator.gather(labels).detach().cpu().numpy()
            all_labels.append(labels)

        completed_steps += 1

    all_logits = np.concatenate(all_logits)
    
    eval_results: Dict[str, float] = {}
    if has_labels:
        all_labels = np.concatenate(all_labels)
        eval_results.update(compute_metrics(eval_metric, (all_logits, all_labels)))
        
    eval_results["completed_steps"] = completed_steps
    eval_results["avg_eval_loss"] = eval_loss / completed_steps

    return eval_results