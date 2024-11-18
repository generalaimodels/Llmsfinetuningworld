from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    """A dataset class that concatenates samples from an input dataset into chunks."""

    def __init__(self, dataset: Dataset, chunk_size: int = 1024):
        """
        Initialize the ConcatDataset.

        Args:
            dataset (Dataset): The input dataset to be chunked.
            chunk_size (int): The size of each chunk. Defaults to 1024.

        Raises:
            ValueError: If chunk_size is not a positive integer.
        """
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples: List[Dict[str, Any]] = []
        self.preprocess_dataset()

    def preprocess_dataset(self) -> None:
        """
        Preprocess the dataset by chunking it into samples.

        This method uses a generator to process the dataset in a memory-efficient manner.
        """
        buffer = defaultdict(list)
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            try:
                for key, value in sample.items():
                    buffer[key].extend(value)

                while all(len(v) >= self.chunk_size for v in buffer.values()):
                    self.samples.append({k: v[:self.chunk_size] for k, v in buffer.items()})
                    buffer = {k: v[self.chunk_size:] for k, v in buffer.items()}

            except (KeyError, TypeError, AttributeError) as e:
                print(f"Error processing sample: {e}")
                continue

        # Handle remaining data
        if any(buffer.values()):
            min_length = min(len(v) for v in buffer.values())
            self.samples.append({k: v[:min_length] for k, v in buffer.items()})

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: The sample at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        if 0 <= idx < len(self.samples):
            return self.samples[idx]
        raise IndexError("Index out of range")

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

import logging
from typing import List, Dict, Any, Iterable
from torch.utils.data import Dataset
from tqdm import tqdm

# Configure logging for error handling and debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConcatDataset(Dataset):
    def __init__(self, dataset: Iterable[Dict[str, List[Any]]], chunk_size: int = 1024):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []

        self._preprocess_dataset()

    def _preprocess_dataset(self) -> None:
        """Preprocess and chunk the input dataset."""
        buffer: Dict[str, List[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "label_g": []
        }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            try:
                # Validate that sample contains all expected keys
                missing_keys = set(buffer.keys()) - sample.keys()
                if missing_keys:
                    raise ValueError(f"Sample is missing keys: {missing_keys}")

                # Concatenate new data into the buffer
                for key in buffer:
                    buffer[key].extend(sample[key])

                # Chunk the buffer if it's above the desired size
                while len(buffer["input_ids"]) >= self.chunk_size:
                    self.samples.append({key: buffer[key][:self.chunk_size] for key in buffer})
                    for key in buffer:
                        buffer[key] = buffer[key][self.chunk_size:]
                        
            except ValueError as e:
                logger.warning(f"Skipping sample due to error: {e}")
            except KeyError as e:
                logger.error(f"Key error for key '{e}' in sample {sample}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error processing sample: {e}", exc_info=True)

    def __getitem__(self, idx: int) -> Dict[str, List[Any]]:
        """Retrieve a specific chunk of data by index."""
        try:
            return self.samples[idx]
        except IndexError:
            logger.error(f"Index {idx} out of bounds for dataset with length {len(self.samples)}")
            raise

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)
