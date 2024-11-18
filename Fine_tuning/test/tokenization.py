from typing import List, Dict, Any
import copy

class TokenizerConfig:
    """
    Configuration class for the tokenizer, handling max length and validation.
    """
    def __init__(self, max_length: int, tokenizer: Any) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        
        required_methods = ["encode", "decode", "bos_token", "eos_token", "pad_token_id"]
        missing_methods = [method for method in required_methods if not hasattr(tokenizer, method)]
        
        if missing_methods:
            raise TypeError(f"The tokenizer must have methods/attributes: {', '.join(missing_methods)}")
        
        self.max_length = max_length
        self.tokenizer = tokenizer


class PromptTemplate:
    """
    Handles prompt templates for construction and formatting of tokenized prompts.
    """
    def __init__(
        self,
        template: str,
        input_variables: List[str],
        config: TokenizerConfig
    ) -> None:
        if not template or not isinstance(template, str):
            raise ValueError("The template must be a non-empty string.")
        self.template = template
        self.input_variables = input_variables
        self.config = config

    def format(self, **kwargs) -> str:
        """
        Formats the prompt using the specified input variables.

        :param kwargs: Dynamic arguments matching input_variables.
        :return: A formatted string.
        :raises KeyError: If required input variables are not provided.
        """
        try:
            formatted = self.template.format(
                **{k: kwargs[k] for k in self.input_variables}
            )
            return formatted
        except KeyError as e:
            raise KeyError(f"Missing required input variable for formatting: {e}")

    def tokenize_and_add_labels(
        self,
        batch: Dict[str, List[str]]
    ) -> Dict[str, List[List[int]]]:
        """
        Tokenizes prompts and targets into input IDs, attention masks, and labels.

        :param batch: A dictionary containing 'prompt' and 'target' lists.
        :return: A dictionary with tokenized data and attention information.
        :raises ValueError: If batch data is inconsistent.
        """
        if 'prompt' not in batch or 'target' not in batch:
            raise ValueError("Batch must contain 'prompt' and 'target' lists.")
        
        if len(batch['prompt']) != len(batch['target']):
            raise ValueError("The length of 'prompt' and 'target' lists must be equal.")

        input_ids, attention_masks, labels = [], [], []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            try:
                encoded_prompt = self.config.tokenizer.encode(
                    self.config.tokenizer.bos_token + prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.config.max_length // 2
                )
                available_length_for_target = self.config.max_length - len(encoded_prompt)

                encoded_target = self.config.tokenizer.encode(
                    target + self.config.tokenizer.eos_token,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=available_length_for_target
                )

                combined = encoded_prompt + encoded_target
                padding_length = self.config.max_length - len(combined)

                input_id = combined + [self.config.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * len(combined) + [0] * padding_length
                label = [-100] * len(encoded_prompt) + encoded_target + [-100] * padding_length

                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                labels.append(label)
            except Exception as e:
                raise RuntimeError(
                    f"Error during tokenization and label addition: {e}. " +
                    "See details above for debugging."
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "label_g": copy.deepcopy(input_ids),
        }
    


import copy
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration class for the tokenizer, handling max length and validation."""
    max_length: int
    tokenizer: Any

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if not hasattr(self.tokenizer, "encode") or not hasattr(self.tokenizer, "decode"):
            raise TypeError("The tokenizer must have `encode` and `decode` methods.")
        
        required_attributes = ["bos_token", "eos_token", "pad_token_id"]
        missing_attributes = [attr for attr in required_attributes if not hasattr(self.tokenizer, attr)]
        if missing_attributes:
            raise AttributeError(f"Tokenizer is missing required attributes: {', '.join(missing_attributes)}")


class PromptTemplate:
    """Handles prompt templates for construction and formatting of tokenized prompts."""

    def __init__(
        self,
        template: str,
        input_variables: List[str],
        config: TokenizerConfig
    ) -> None:
        if not template or not isinstance(template, str):
            raise ValueError("The template must be a non-empty string.")
        self.template = template
        self.input_variables = input_variables
        self.config = config

    def format(self, **kwargs) -> str:
        """
        Formats the prompt using the specified input variables.

        :param kwargs: Dynamic arguments matching input_variables.
        :return: A formatted string.
        :raises KeyError: If required input variables are not provided.
        """
        try:
            return self.template.format(**{k: kwargs[k] for k in self.input_variables})
        except KeyError as e:
            raise KeyError(f"Missing required input variable for formatting: {e}")

    def tokenize_and_add_labels(
        self,
        batch: Dict[str, List[str]]
    ) -> Dict[str, List[List[int]]]:
        """
        Tokenizes prompts and targets into input IDs, attention masks, and labels.

        :param batch: A dictionary containing 'prompt' and 'target' lists.
        :return: A dictionary with tokenized data and attention information.
        :raises ValueError: If input data is inconsistent or missing.
        :raises RuntimeError: If tokenization fails.
        """
        if "prompt" not in batch or "target" not in batch:
            raise ValueError("Batch must contain both 'prompt' and 'target' keys.")
        if len(batch["prompt"]) != len(batch["target"]):
            raise ValueError("The number of prompts and targets must be equal.")

        input_ids, attention_masks, labels = [], [], []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            try:
                encoded_prompt = self._encode_text(
                    self.config.tokenizer.bos_token + prompt,
                    max_length=self.config.max_length // 2
                )
                encoded_target = self._encode_text(
                    target + self.config.tokenizer.eos_token,
                    max_length=self.config.max_length - len(encoded_prompt)
                )

                combined = encoded_prompt + encoded_target
                padding_length = self.config.max_length - len(combined)

                input_id = combined + [self.config.tokenizer.pad_token_id] * padding_length
                attention_mask = [1] * len(combined) + [0] * padding_length
                label = (
                    [-100] * len(encoded_prompt) +
                    encoded_target +
                    [-100] * padding_length
                )

                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                labels.append(label)
            except Exception as e:
                raise RuntimeError(f"Error during tokenization and label addition: {str(e)}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "label_g": copy.deepcopy(input_ids),
        }

    def _encode_text(self, text: str, max_length: int) -> List[int]:
        """
        Encodes text using the tokenizer with proper error handling.

        :param text: The text to encode.
        :param max_length: Maximum length for encoding.
        :return: List of encoded token IDs.
        :raises RuntimeError: If encoding fails.
        """
        try:
            return self.config.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to encode text: {str(e)}")


def create_prompt_template(
    template: str,
    input_variables: List[str],
    tokenizer: Any,
    max_length: int
) -> PromptTemplate:
    """
    Factory function to create a PromptTemplate instance with proper configuration.

    :param template: The prompt template string.
    :param input_variables: List of input variable names.
    :param tokenizer: The tokenizer object.
    :param max_length: Maximum length for tokenization.
    :return: An instance of PromptTemplate.
    """
    config = TokenizerConfig(max_length=max_length, tokenizer=tokenizer)
    return PromptTemplate(template=template, input_variables=input_variables, config=config)