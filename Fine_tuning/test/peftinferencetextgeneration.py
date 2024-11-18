import csv
import logging
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self, peft_model_id: str, base_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.peft_model_id = peft_model_id
        self.base_path = base_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        try:
            config = PeftConfig.from_pretrained(self.peft_model_id)
            model = AutoModelForCausalLM.from_pretrained(self.base_path)
            self.model = PeftModel.from_pretrained(model, self.peft_model_id).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_path)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], max_length: int = 512) -> List[str]:
        try:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Batch text generation failed: {str(e)}")
            return [""] * len(prompts)

def process_dataset(pipeline: InferencePipeline, dataset: Dict[str, Any], batch_size: int = 16) -> List[Dict[str, str]]:
    results = []
    prompts = [item["prompt"] for item in dataset["train"]]
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = pipeline.generate_batch(batch_prompts)
        
        for prompt, response in zip(batch_prompts, batch_responses):
            results.append({"Query": prompt, "Response": response})
    
    return results

def save_to_csv(data: List[Dict[str, str]], output_file: str) -> None:
    try:
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {str(e)}")

def main():
    peft_model_id = "/scratch/hemanth/peft_weights"
    base_path = "/scratch/hemanth/model/models--microsoft--Phi-3.5-mini-instruct/snapshots/ccf028fc8e1b3ab750a7c55b22792f57ba69f216"
    output_file = "inference_results2.csv"
    batch_size = 16  # Adjust based on your GPU memory and model size

    try:
        pipeline = InferencePipeline(peft_model_id, base_path)
        pipeline.load_model()

        dataset = load_dataset("fka/awesome-chatgpt-prompts")
        results = process_dataset(pipeline, dataset, batch_size)
        save_to_csv(results, output_file)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()