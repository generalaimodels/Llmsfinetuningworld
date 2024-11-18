import logging
import copy
from typing import Dict, List
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer


from .data_loader import DatasetLoader
from cfinetuning.cconfig import DataConfig,DatasetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        return self.template.format(**{k: kwargs.get(k, '') for k in self.input_variables})

class DatasetProcessor:
    def __init__(
        self,
        dataloader_config:  DataConfig,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        prompt_template: PromptTemplate,
    ):
        self.config = dataset_config
        self.datasetloader_config=dataloader_config
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        

    def load_and_split_dataset(self) -> DatasetDict:
        try:
            dataset = DatasetLoader(config=self.datasetloader_config).load()
            return dataset.train_test_split(
                test_size=self.config.eval_ratio + self.config.test_ratio,
                shuffle=True,
                seed=42,
            )
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def validate_columns(self, dataset: Dataset):
        all_columns = self.config.input_columns + [self.config.target_column]
        missing_columns = [col for col in all_columns if col not in dataset.column_names]
        if missing_columns:
            raise ValueError(f"Missing columns in dataset: {missing_columns}")

    def apply_prompt_template(self, batch: Dict[str, List]) -> Dict[str, List]:
        prompts = []
        targets = []
        for i in range(len(batch[self.config.input_columns[0]])):
            input_dict = {col: batch[col][i] for col in self.config.input_columns}
            prompts.append(self.prompt_template.format(**input_dict))
            targets.append(batch[self.config.target_column][i])
        return {"prompt": prompts, "target": targets}

    def tokenize_and_add_labels(self, batch: Dict[str, List]) -> Dict[str, List]:
        input_ids = []
        attention_masks = []
        labels = []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            encoded_prompt = self.tokenizer.encode(
                self.tokenizer.bos_token + prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.max_length // 2,
            )
            encoded_target = self.tokenizer.encode(
                target + self.tokenizer.eos_token,
                add_special_tokens=False,
                truncation=True,
                max_length=self.config.max_length - len(encoded_prompt),
            )
            
            combined = encoded_prompt + encoded_target
            padding_length = self.config.max_length - len(combined)

            input_id = combined + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(combined) + [0] * padding_length
            label = [-100] * len(encoded_prompt) + encoded_target + [-100] * padding_length

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }

    def process_dataset(self) -> DatasetDict:
        try:
            dataset_dict = self.load_and_split_dataset()
            for split in dataset_dict:
                self.validate_columns(dataset_dict[split])
                dataset_dict[split] = (
                    dataset_dict[split]
                    .map(
                        self.apply_prompt_template,
                        batched=True,
                        batch_size=self.config.batch_size,
                        remove_columns=dataset_dict[split].column_names,
                    )
                    .map(
                        self.tokenize_and_add_labels,
                        batched=True,
                        batch_size=self.config.batch_size,
                        remove_columns=["prompt", "target"],
                    )
                )
            return dataset_dict
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

