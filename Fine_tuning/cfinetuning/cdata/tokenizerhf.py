
import os
from typing import List, Literal, Optional, Sequence, TypedDict, Union, Dict, Any
import pandas as pd
import logging
from transformers import PreTrainedTokenizer, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

Dialog = Sequence[Message]

class Tokenizer:
    def __init__(self, tokenizer: Union[str, PreTrainedTokenizer]):
        try:
            if isinstance(tokenizer, str):
                self.model = AutoTokenizer.from_pretrained(tokenizer)
            else:
                self.model = tokenizer
            
            # Set padding token if it doesn't exist
            if self.model.pad_token is None:
                self.model.pad_token = self.model.eos_token
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

        logger.info(f"Loaded tokenizer: {self.model.__class__.__name__}")

        self.n_words: int = len(self.model.vocab)
        self.bos_id: Optional[int] = self.model.bos_token_id
        self.eos_id: Optional[int] = self.model.eos_token_id
        self.pad_id: Optional[int] = self.model.pad_token_id
        self.stop_tokens: set = {self.eos_id} if self.eos_id is not None else set()

        logger.info(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}")

    def encode(self, s: str, **kwargs) -> List[int]:
        try:
            return self.model.encode(s, **kwargs)
        except Exception as e:
            logger.error(f"Error encoding string: {e}")
            raise

    def decode(self, t: Sequence[int], **kwargs) -> str:
        try:
            return self.model.decode(t, **kwargs)
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            raise

class ChatFormat:
    def __init__(self, tokenizer: Tokenizer, format_config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.format_config = format_config

    def encode_message(self, message: Message) -> List[int]:
        tokens = []
        role_format = self.format_config.get("role_format", "<|{role}|>")
        content_format = self.format_config.get("content_format", "{content}")
        separator = self.format_config.get("separator", "\n\n")
        eot_token = self.format_config.get("eot_token", "<|eot|>")

        tokens.extend(self.tokenizer.encode(role_format.format(role=message['role'])))
        tokens.extend(self.tokenizer.encode(separator))
        tokens.extend(self.tokenizer.encode(content_format.format(content=message['content'].strip())))
        tokens.extend(self.tokenizer.encode(eot_token))
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        begin_token = self.format_config.get("begin_token", "<|begin_of_text|>")
        tokens.extend(self.tokenizer.encode(begin_token))
        
        for message in dialog:
            tokens.extend(self.encode_message(message))
        
        assistant_start = self.format_config.get("assistant_start", "<|assistant|>")
        tokens.extend(self.tokenizer.encode(assistant_start))
        return tokens

class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]) -> None:
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**{k: kwargs.get(k, '') for k in self.input_variables})

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



# # Example usage
# if __name__ == "__main__":
#     # Load a tokenizer
#     tokenizer = Tokenizer("gpt2")

    # # Define chat format configuration
    # chat_format_config = {
    #     "role_format": "<|{role}|>",
    #     "content_format": "{content}",
    
    # }

    # # Create a ChatFormat instance
    # chat_format = ChatFormat(tokenizer, chat_format_config)
    # t="And what about Germany?"
    # # Example dialog
    # dialog: Dialog = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": t},
    #     {"role": "assistant", "content": "The capital of France is Paris."},
    #     {"role": "user", "content": t},
    # ]

    # # Encode the dialog
    # encoded_dialog = chat_format.encode_dialog_prompt(dialog)

    # print(f"{encoded_dialog}")

    # print("Encoded dialog:", encoded_dialog)
    # print("Decoded dialog:", tokenizer.decode(encoded_dialog))


    # df = pd.DataFrame({
    # 'col1': ['Hello', 'This is', 'A sample'],
    # 'col2': ['world', 'a test', 'dataset'],
    # 'target': ['Greeting', 'Statement', 'Description']
    #  })
    
    # columns: List[str] = ["col1", "col2"]
    # target_column: str = "target"
    # prompt_template = PromptTemplate("Input: {col1} {col2}", ["col1", "col2"])
    # max_length: int = 512
    # batch_size: int = 32

    # try:
    #     result = process_data(df, columns, target_column, prompt_template, max_length, tokenizer, batch_size)

    #     logger.info("Data processing completed successfully.")
    #     logger.info(f"Number of samples: {len(result['input_ids'])}")
    #     logger.info(f"Sample input_ids: {result['input_ids']}...")
    #     logger.info(f"Sample attention_mask: {result['attention_mask']}...")
    #     logger.info(f"Sample labels: {result['labels']}...")
    #     logger.info(f"Sample label_g: {result['label_g']}...")

    #     sample_input = tokenizer.decode(result['input_ids'][0])
    #     sample_label = tokenizer.decode(result['labels'][0])
    #     logger.info(f"Sample decoded input: {sample_input}")
    #     logger.info(f"Sample decoded label: {sample_label}")

    # except Exception as e:
    #     logger.error(f"An error occurred: {e}")