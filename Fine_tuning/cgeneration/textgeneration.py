from typing import List, Optional, Tuple, Dict, Any, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataclasses import dataclass
from enum import Enum

@dataclass
class CompletionPrediction:
    text: str
    logprobs: Optional[List[float]] = None

@dataclass
class ChatPrediction:
    response: str
    logprobs: Optional[List[float]] = None

class Role(Enum):
    HUMAN = "Human"
    ASSISTANT = "Assistant"
    SYSTEM = "System"

class Message:
    def __init__(self, role: Role, content: str):
        self.role = role
        self.content = content

class Dialog(List[Message]):
    pass

class TextGenerator:
    """
    Example:
    # Create an instance
    generator = TextGenerator.from_pretrained("MY_model")
    args={"temperature": 0.7, "top_p": 0.9,"do_sample":True}
    # Text completion example
    prompts = ["Once upon a time", "The quick brown fox"]
    completions = generator.complete(prompts, max_gen_len=50, sampling_params=args)
    for prompt, completion in zip(prompts, completions):
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion.text}\n")
    
    # Chat example
    dialogs = [
        Dialog([
            Message(Role.HUMAN, "Hello, how are you?"),
            Message(Role.ASSISTANT, "I'm doing well, thank you. How can I assist you today?"),
            Message(Role.HUMAN, "Can you tell me a joke?")
        ])
    ]
    chat_responses = generator.chat(dialogs, max_gen_len=100, sampling_params=args)
    for dialog, response in zip(dialogs, chat_responses):
        print("Dialog:")
        for message in dialog:
            print(f"- {message.role.value}: {message.content}")
        print(f"Assistant's response: {response.response}\n")
    
    """
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        sampling_params: Dict[str, Any],
        logprobs: bool = False,
        echo: bool = False
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        try:
            batch_size = len(prompt_tokens)
            max_prompt_len = max(len(seq) for seq in prompt_tokens)
            
            if max_prompt_len + max_gen_len > self.model.config.max_position_embeddings:
                max_gen_len = self.model.config.max_position_embeddings - max_prompt_len
                print(f"Warning: Reduced max_gen_len to {max_gen_len} due to model constraints.")

            padded_prompts = [
                seq + [self.tokenizer.pad_token_id] * (max_prompt_len - len(seq))
                for seq in prompt_tokens
            ]
            attention_masks = [
                [1] * len(seq) + [0] * (max_prompt_len - len(seq))
                for seq in prompt_tokens
            ]

            input_ids = torch.tensor(padded_prompts).to(self.device)
            attention_mask = torch.tensor(attention_masks).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_prompt_len + max_gen_len,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_scores=logprobs,
                    return_dict_in_generate=True,
                    **sampling_params
                )

            generated_sequences = outputs.sequences.tolist()

            if not echo:
                generated_sequences = [
                    seq[len(prompt):] for seq, prompt in zip(generated_sequences, prompt_tokens)
                ]

            if logprobs:
                log_probs = torch.stack(outputs.scores, dim=1).log_softmax(-1)
                token_log_probs = [
                    log_probs[i, range(len(seq)), seq].tolist()
                    for i, seq in enumerate(generated_sequences)
                ]
                return generated_sequences, token_log_probs

            return generated_sequences, None

        except Exception as e:
            raise RuntimeError(f"Error during text generation: {str(e)}")

    def complete(
        self,
        prompts: List[str],
        max_gen_len: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        echo: bool = False
    ) -> List[CompletionPrediction]:
        try:
            if max_gen_len is None:
                max_gen_len = self.model.config.max_position_embeddings - 1

            if sampling_params is None:
                sampling_params = {"temperature": 0.6, "top_p": 0.9, "do_sample": True}

            tokenized_prompts = [
                self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).tolist()
                for prompt in prompts
            ]
            generated_tokens, token_log_probs = self.generate(
                tokenized_prompts,
                max_gen_len,
                sampling_params,
                logprobs,
                echo
            )

            completions = []
            for i, tokens in enumerate(generated_tokens):
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                logprob = token_log_probs[i] if logprobs else None
                completions.append(CompletionPrediction(text=text, logprobs=logprob))

            return completions

        except Exception as e:
            raise RuntimeError(f"Error during text completion: {str(e)}")

    def chat(
        self,
        dialogs: List[Dialog],
        max_gen_len: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        format_func: Optional[callable] = None
    ) -> List[ChatPrediction]:
        try:
            if max_gen_len is None:
                max_gen_len = self.model.config.max_position_embeddings - 1

            if sampling_params is None:
                sampling_params = {"temperature": 0.6, "top_p": 0.9, "do_sample": True}

            if format_func is None:
                format_func = self.default_format_dialog

            chat_predictions = []
            for dialog in dialogs:
                formatted_dialog = format_func(dialog)
                tokenized_dialog = self.tokenizer.encode(formatted_dialog, return_tensors="pt").squeeze(0).tolist()

                generated_tokens, token_log_probs = self.generate(
                    [tokenized_dialog],
                    max_gen_len,
                    sampling_params,
                    logprobs,
                    echo=False
                )

                response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                logprob = token_log_probs[0] if logprobs else None
                chat_predictions.append(ChatPrediction(response=response, logprobs=logprob))

            return chat_predictions

        except Exception as e:
            raise RuntimeError(f"Error during chat generation: {str(e)}")

    @staticmethod
    def default_format_dialog(dialog: Dialog) -> str:
        formatted = ""
        for message in dialog:
            formatted += f"{message.role.value}: {message.content}\n"
        formatted += f"{Role.ASSISTANT.value}: "
        return formatted

    def batch_chat(
        self,
        dialogs: List[Dialog],
        max_gen_len: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        logprobs: bool = False,
        format_func: Optional[callable] = None,
        batch_size: int = 4
    ) -> List[ChatPrediction]:
        try:
            all_predictions = []
            for i in range(0, len(dialogs), batch_size):
                batch_dialogs = dialogs[i:i+batch_size]
                batch_predictions = self.chat(
                    batch_dialogs,
                    max_gen_len,
                    sampling_params,
                    logprobs,
                    format_func
                )
                all_predictions.extend(batch_predictions)
            return all_predictions
        except Exception as e:
            raise RuntimeError(f"Error during batch chat generation: {str(e)}")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ) -> 'TextGenerator':
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            return cls(model, tokenizer, device)
        except Exception as e:
            raise RuntimeError(f"Error loading pretrained model: {str(e)}")

    def to(self, device: Union[str, torch.device]) -> 'TextGenerator':
        self.device = torch.device(device)
        self.model.to(self.device)
        return self
