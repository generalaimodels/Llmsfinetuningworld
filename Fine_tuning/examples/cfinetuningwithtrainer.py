# coding=utf-8

import sys
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, Optional

import yaml
import plotly.graph_objects as go
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from datasets import DatasetDict
from evaluate import load


# Append the path for custom modules
sys.path.append(str(Path(__file__).resolve().parents[3]))

# Import custom modules
from cfinetuning import (
    DataConfig,
    DatasetConfig,
    DatasetProcessor,
    ModelConfig,
    ModelLoader,
    PromptTemplate,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE_PATH = Path(r"E:\LLMS\Fine-tuning\LlmsComponents\Fine_tuning\examples\ccongfig.yml")
DEFAULT_SAVE_PATH = 'best_weights'
OUTPUT_DIR = 'output'
LOGGING_DIR = 'logs'

def load_config(file_path: Path) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except (yaml.YAMLError, FileNotFoundError) as e:
        logger.error(f"Error loading config: {e}")
        raise

def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def plot_training_progress(trainer: Trainer):
    """Plot training progress using Plotly."""
    try:
        history = trainer.state.log_history
        train_loss = [log['loss'] for log in history if 'loss' in log]
        eval_loss = [log['eval_loss'] for log in history if 'eval_loss' in log]
        eval_accuracy = [log['eval_accuracy'] for log in history if 'eval_accuracy' in log]

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_loss, mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(y=eval_loss, mode='lines', name='Evaluation Loss'))
        fig.add_trace(go.Scatter(y=eval_accuracy, mode='lines', name='Evaluation Accuracy'))
        
        fig.update_layout(title='Training Progress', xaxis_title='Steps', yaxis_title='Metrics')
        fig.write_html("training_progress.html")
        logger.info("Training progress plot saved as training_progress.html")
    except Exception as e:
        logger.error(f"Error plotting training progress: {e}")

config_data = load_config(CONFIG_FILE_PATH)

logger.info("Extract configuration arguments using nested get method for better readability ...")

data_loader_args = config_data.get('config1', {}).get('datasetconfig', {}).get('DataConfig', {})
data_preprocess_args = config_data.get('config1', {}).get('datasetconfig', {}).get('DatasetConfig', {})
prompt_template_args = config_data.get('config1', {}).get('promptconfig', {}).get('PromptTemplate', {})
model_loader_args = config_data.get('config1', {}).get('modelconfig', {}).get('ModelConfig', {})
TrainingArguments_loader_args=config_data.get('config1', {}).get('trainingconfig', {}).get('CTrainingArguments', {})

logger.info("Initialize configuration objects ...")

data_config = DataConfig(**data_loader_args)  # type: ignore
dataset_config = DatasetConfig(**data_preprocess_args)  # type: ignore
prompt_template = PromptTemplate(**prompt_template_args)  # type: ignore
model_config = ModelConfig(**model_loader_args)  # type: ignore
training_args=TrainingArguments(**TrainingArguments_loader_args)

logger.info("Initialize and load model and tokenizer ...")

model_loader = ModelLoader(model_config)
model, tokenizer = model_loader.load_model_and_tokenizer()




if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_configuration = model_loader.get_config()
logger.info(f"Model config file detail:{model_configuration}")


logger.info("Initialize dataset processor with configurations...")

dataset_processor = DatasetProcessor(
    dataloader_config=data_config,
    dataset_config=dataset_config,
    prompt_template=prompt_template,
    tokenizer=tokenizer
)
logger.info(" Process datasets...")
processed_datasets: DatasetDict = dataset_processor.process_dataset()

logger.info("Define a data collator for dynamic padding...")

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding='longest',
    max_length=dataset_config.max_length  # type: ignore
)


class SaveBestModelCallback(TrainerCallback):
    """A custom callback to save the best model weights based on evaluation metrics."""
    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.best_metric: Optional[float] = None

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        metrics = kwargs.pop("metrics", {})
        eval_accuracy = metrics.get("eval_accuracy")
        if eval_accuracy is not None and (self.best_metric is None or eval_accuracy > self.best_metric):
            self.best_metric = eval_accuracy
            logger.info(f"New best metric: {self.best_metric}. Saving model.")
            self.save_model_weights()
            control.should_save = True

    def save_model_weights(self) -> None: # saves the model weights
        model.save_pretrained(self.save_path)
        tokenizer.save_pretrained(self.save_path)


metric = load("accuracy")
logger.info(f"Define the evaluation metric {metric.__dict__ } ...")

def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = logits.argmax(-1)

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


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets['train'],
    eval_dataset=processed_datasets['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[SaveBestModelCallback(save_path=DEFAULT_SAVE_PATH)]
)

# Train and evaluate the model
try:
    train_result = trainer.train()
    trainer.evaluate()

    # Save training metrics
    metrics = train_result.metrics
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    plot_training_progress(trainer)
    logger.info("Training and evaluation completed successfully.")

except Exception as e:
    logger.error(f"Training failed: {e}")