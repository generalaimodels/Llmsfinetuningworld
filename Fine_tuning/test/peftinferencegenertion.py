from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict

# Set paths to model and configuration
PEFT_MODEL_ID = "/scratch/hemanth/peft_weights"
BASE_PATH = "/scratch/hemanth/model/models--microsoft--Phi-3.5-mini-instruct/snapshots/ccf028fc8e1b3ab750a7c55b22792f57ba69f216"

# Load configuration and model
try:
    config = PeftConfig.from_pretrained(PEFT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_PATH)
    model = PeftModel(model, config)
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")

# Ensure model is set to evaluation mode
model.eval()

# Check CUDA availability and load model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class PromptsDataset(Dataset):
    def __init__(self, prompts: List[str], max_length: int = 512):
        self.prompts = prompts
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs
def generate_batch_texts(batch: Dict[str, torch.Tensor], max_length: int = 512) -> List[str]:
    batch = {key: val.to(device) for key, val in batch.items()}
    
    # Ensure all tensors in the batch have the same size
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    
    if attention_mask is not None and attention_mask.size() != input_ids.size():
        attention_mask = attention_mask[:, :input_ids.size(1)]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def generate_responses_from_dataset(batch_size: int = 4, max_length: int = 512) -> pd.DataFrame:
    try:
        dataset = load_dataset("fka/awesome-chatgpt-prompts")
        queries = [example['prompt'] for example in dataset['train']]
        prompts_dataset = PromptsDataset(queries, max_length=max_length)
        data_loader = DataLoader(prompts_dataset, batch_size=batch_size, shuffle=False)

        all_responses = []
        for batch in data_loader:
            responses = generate_batch_texts(batch, max_length=max_length)
            all_responses.extend(responses)

        # Ensure all queries and responses are paired
        data = [{"Query": query, "Response": response} for query, response in zip(queries, all_responses)]

        return pd.DataFrame(data)
    except Exception as e:
        raise RuntimeError(f"Failed to process dataset: {e}")

def main():
    """
    Main function to run the inference pipeline and save the results to a CSV file.
    """
    try:
        # Generate responses
        df = generate_responses_from_dataset()
        
        # Save results to CSV
        output_file = "generated_responses.json"
        df.to_json(output_file, index=False)
        print(f"Results saved to {output_file}")
    except RuntimeError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()