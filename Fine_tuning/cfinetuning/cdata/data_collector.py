import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from itertools import islice
import logging
from typing import Dict, List, Any, Iterator, Optional
from collections import defaultdict
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=1024):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "label_g":[]
            }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class ConcatDatasetbatch(Dataset):
    def __init__(self, dataset, chunk_size=1024, batch_size=32):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k, v in buffer.items()}

            while len(next(iter(buffer.values()))) >= self.chunk_size * self.batch_size:
                self.samples.append({
                    k: [v[i:i + self.chunk_size] for i in range(0, len(v), self.chunk_size)][:self.batch_size] 
                    for k, v in buffer.items()
                })
                buffer = {k: v[self.chunk_size * self.batch_size:] for k, v in buffer.items()}

        # Handle any remaining samples in the buffer
        if any(len(v) > 0 for v in buffer.values()):
            self.samples.append({
                k: [v[i:i + self.chunk_size] for i in range(0, len(v), self.chunk_size)][:self.batch_size]
                for k, v in buffer.items()
            })

    def __getitem__(self, idx):
        if idx is not None:
           return self.samples[idx]
        else:
            raise TypeError(f"Index must be an integer, not {type(idx).__name__}")

    def __len__(self):
        return len(self.samples)



class Datacollector(IterableDataset):
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