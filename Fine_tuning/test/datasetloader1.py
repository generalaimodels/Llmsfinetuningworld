import logging
from typing import Dict, List, Any, Iterator, Optional
from collections import defaultdict
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader(IterableDataset):
    def __init__(
        self,
        dataset: Dict[str, List[Any]],
        batch_size: int,
        chunk_size: int = 1000,
        shuffle: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the DatasetLoader.

        Args:
            dataset (Dict[str, List[Any]]): The input dataset.
            batch_size (int): The size of each batch.
            chunk_size (int): The size of each chunk for processing.
            shuffle (bool): Whether to shuffle the dataset.
            device (Optional[torch.device]): The device to load tensors to.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.device = device or torch.device("cpu")
        self.keys = list(dataset.keys())
        self.dataset_size = len(next(iter(dataset.values())))
        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """Validate the input dataset for consistency."""
        if not all(len(v) == self.dataset_size for v in self.dataset.values()):
            raise ValueError("All dataset elements must have the same length.")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over the dataset, yielding batches."""
        chunk_generator = self._chunk_generator()
        buffer = defaultdict(list)
        buffer_size = 0

        for chunk in chunk_generator:
            for key, values in chunk.items():
                buffer[key].extend(values)
            buffer_size += len(next(iter(chunk.values())))

            while buffer_size >= self.batch_size:
                batch = {k: buffer[k][:self.batch_size] for k in self.keys}
                for key in self.keys:
                    buffer[key] = buffer[key][self.batch_size:]
                buffer_size -= self.batch_size

                try:
                    yield {k: torch.tensor(v, device=self.device) for k, v in batch.items()}
                except Exception as e:
                    logger.error(f"Error creating tensor batch: {e}")
                    continue

        # Handle remaining data in buffer
        if buffer_size > 0:
            batch = {k: buffer[k] for k in self.keys}
            try:
                yield {k: torch.tensor(v, device=self.device) for k, v in batch.items()}
            except Exception as e:
                logger.error(f"Error creating final tensor batch: {e}")

    def _chunk_generator(self) -> Iterator[Dict[str, List[Any]]]:
        """Generate chunks of the dataset."""
        indices = list(range(self.dataset_size))
        if self.shuffle:
            torch.manual_seed(42)  # For reproducibility
            indices = torch.randperm(self.dataset_size).tolist()

        for i in range(0, self.dataset_size, self.chunk_size):
            chunk_indices = indices[i:i+self.chunk_size]
            chunk = {k: [self.dataset[k][j] for j in chunk_indices] for k in self.keys}
            yield chunk

    def __len__(self) -> int:
        """Return the number of batches in the dataset."""
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

# # Example usage
# if __name__ == "__main__":
#     # Sample dataset
#     dataset = {
#         "input_ids": list(range(10000)),
#         "attention_mask": list(range(10000)),
#         "labels": list(range(10000)),
#         "label_g": list(range(10000))
#     }

#     loader = DatasetLoader(dataset, batch_size=1000, chunk_size=1000, shuffle=True)
    
#     for batch in tqdm(loader, total=len(loader)):
#         print(batch["labels"])
      
