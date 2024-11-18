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
    def generate_text(self, prompt: str, max_length: int = 512) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return ""

def process_dataset(pipeline: InferencePipeline, dataset: Dict[str, Any]) -> List[Dict[str, str]]:
    results = []
    for item in tqdm(dataset["train"], desc="Processing queries"):
        query = item["prompt"]
        response = pipeline.generate_text(query)
        results.append({"Query": query, "Response": response})
    return results

def save_to_csv(data: List[Dict[str, str]], output_file: str) -> None:
    try:
        df = pd.DataFrame(data)
        df.to_json(output_file, index=False, quoting=csv.QUOTE_ALL)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {str(e)}")

def main():
    peft_model_id = "/scratch/hemanth/peft_weights"
    base_path = " "
    output_file = "inference_results.json"

    try:
        pipeline = InferencePipeline(peft_model_id, base_path)
        pipeline.load_model()

        dataset = load_dataset("fka/awesome-chatgpt-prompts")
        results = process_dataset(pipeline, dataset)
        save_to_csv(results, output_file)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()