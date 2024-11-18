import os
from typing import (
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    Union,
    Dict, 
    Any
)
import pandas as pd
import logging
from transformers import PreTrainedTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the AutoTokenizer from transformers.
    """

    def __init__(self, tokenizer:PreTrainedTokenizer):
        """
        Initializes the Tokenizer with a pre-trained model.

        Args:
            model_name (str): The name or path of the pre-trained model.
        """
        try:
            self.model = tokenizer
        except Exception as e:
            logger.error(f"Failed to load AutoTokenizer: {e}")
            raise

        logger.info("Loaded AutoTokenizer for model...")

        self.n_words: int = len(self.model.vocab)
        self.bos_id: int = self.model.bos_token_id
        self.eos_id: int = self.model.eos_token_id
        self.pad_id: int = self.model.pad_token_id
        self.stop_tokens: set = {self.eos_id}

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def encode(
        self,
        s: str,
        *,
        bos: bool = False,
        eos: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            max_length (Optional[int]): Maximum length of the encoded sequence.
            truncation (bool): Whether to truncate sequences exceeding max_length.
            padding (bool): Whether to pad sequences to max_length.

        Returns:
            List[int]: A list of token IDs.
            
        Example :    # Create a ChatFormat instance
            chat_format = ChatFormat(tokenizer)
        
            # Example dialog
            dialog: Dialog = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "And what about Germany?"},
            ]
        
            # Encode the dialog
            encoded_dialog = chat_format.encode_dialog_prompt(dialog)
        
            print("Encoded dialog:", encoded_dialog)
            print("Decoded dialog:", tokenizer.decode(encoded_dialog))
        """
        if not isinstance(s, str):
            raise TypeError("Input must be a string")

        try:
            encoded = self.model.encode(
                s,
                add_special_tokens=False,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
            )
            
            if bos:
                encoded = [self.bos_id] + encoded
            if eos:
                encoded = encoded + [self.eos_id]
            
            return encoded
        except Exception as e:
            logger.error(f"Error encoding string: {e}")
            raise

    def decode(self, t: Sequence[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (Sequence[int]): The sequence of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        try:
            return self.model.decode(t, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            raise


class ChatFormat:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.extend(self.tokenizer.encode(f"<|{message['role']}|>", bos=False, eos=False))
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.extend(self.tokenizer.encode("<|eot|>", bos=False, eos=False))
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.extend(self.tokenizer.encode("<|begin_of_text|>", bos=True, eos=False))
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]) -> None:
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**{k: kwargs.get(k, '') for k in self.input_variables})

class TokenizerTemplate:
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """

             
        """
        try:
            self.model: tokenizer
            # Set padding token
            if self.model.pad_token is None:
                self.model.pad_token = self.model.eos_token
                logger.info(f"Set padding token to: {self.model.pad_token}")
        except Exception as e:
            logger.error(f"Failed to load AutoTokenizer: {e}")
            raise

        logger.info(f"Loaded AutoTokenizer for model")

        self.n_words: int = len(self.model.vocab)
        self.bos_id: int = self.model.bos_token_id
        self.eos_id: int = self.model.eos_token_id
        self.pad_id: int = self.model.pad_token_id
        self.stop_tokens: set = {self.eos_id}

    def encode(self, text: str, **kwargs) -> List[int]:
        return self.model.encode(text, **kwargs)

    def decode(self, tokens: List[int]) -> str:
        return self.model.decode(tokens, skip_special_tokens=True)

def process_data(
    data: pd.DataFrame,
    columns: List[str],
    target_column: str,
    prompt_template: PromptTemplate,
    max_length: int,
    tokenizer: Tokenizer,
    batch_size: int
) -> Dict[str, Any]:
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not all(col in data.columns for col in columns + [target_column]):
            raise ValueError("Specified columns not found in the DataFrame")

        input_texts = data[columns].apply(
            lambda row: prompt_template.format(**row.to_dict()), axis=1
        )
        labels = data[target_column].tolist()

        input_ids = []
        attention_masks = []

        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i+batch_size].tolist()
            encoded = tokenizer.model(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids.extend(encoded['input_ids'].tolist())
            attention_masks.extend(encoded['attention_mask'].tolist())

        label_ids = tokenizer.model(
            labels,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )['input_ids'].tolist()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": label_ids,
            "label_g": input_ids.copy()
        }

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

