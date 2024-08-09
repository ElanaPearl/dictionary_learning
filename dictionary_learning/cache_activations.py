"""Module for caching activations of a model on a dataset.

To use this module, you need to implement a class that implements the PretokenizedDataSource interface,
load a pre-trained model from the transformers library, and call the calculate_and_save_activations function
via calculate_and_save_activations(model, data_source)
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel


class PretokenizedDataSource(ABC):
    """Interface for a data source that provides pretokenized sequences."""

    @property
    @abstractmethod
    def total_samples(self) -> int:
        """
        The total number of samples in the data source.

        Returns:
            int: The total number of samples.
        """
        pass

    @abstractmethod
    def get_batch(self, start: int, size: int) -> List[str]:
        """
        Retrieve a batch of data.

        Args:
            start (int): The starting index of the batch.
            size (int): The size of the batch to retrieve.

        Returns:
            List[str]: A list of strings representing the batch of data.
        """
        pass


class StreamingDataset(Dataset):
    """
    A PyTorch Dataset that streams data from a PretokenizedDataSource.
    """

    def __init__(self, data_source: PretokenizedDataSource, batch_size: int):
        self.data_source = data_source
        self.batch_size = batch_size
        self.current_position = 0

    def __len__(self):
        return (self.data_source.total_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx: int) -> List[str]:
        batch = self.data_source.get_batch(self.current_position, self.batch_size)
        self.current_position += len(batch)
        return batch.tolist()

    def reset(self):
        self.current_position = 0


def get_activations(
    model: PreTrainedModel,
    batch: list[str],
    tokenizer_kwargs: dict = {"padding": True, "truncation": True, "max_length": 1024},
) -> torch.Tensor:
    """Get the activations of the last hidden layer of the model for a batch of sequences."""
    inputs = model.tokenizer(
        batch,
        return_tensors="pt",
        **tokenizer_kwargs,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        hidden_states = model(**inputs, return_dict=True).last_hidden_state

    # Create attention mask that ignores padding tokens (attention_mask=0) and special tokens (input_ids=0,1,2)
    token_mask = inputs["attention_mask"] * (inputs["input_ids"] > 2)

    # Apply token mask (also remove first and last tokens at beginning of mask) and concatenate all activations
    hidden_states = hidden_states[token_mask != 0]

    return hidden_states


def calculate_and_save_activations(
    model: PreTrainedModel,
    data_source: PretokenizedDataSource,
    output_dir: Path = Path("activation_cache"),
    batch_size: int = 32,
    tokenizer_kwargs: dict = {"padding": True, "truncation": True, "max_length": 1024},
    save_dtype: str = "float32",
) -> tuple[int, int]:
    """Calculate and save the activations of a model on a dataset."""

    # Create output director
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    activations_path = output_dir / "activations.dat"
    metadata_path = output_dir / "metadata.json"

    data_loader = DataLoader(
        dataset=StreamingDataset(data_source, batch_size=batch_size),
        batch_size=1,
        shuffle=False,
        collate_fn=custom_str_collate,
    )

    # First pass: count total tokens to determine memmap size
    max_tokens_per_seq = (
        tokenizer_kwargs.get("max_length", 1024) - 2
    )  # Subtract 2 for start/end tokens
    total_tokens = 0
    for batch in tqdm(data_loader, desc="Counting tokens", total=len(data_loader)):
        total_tokens += sum([min(len(x), max_tokens_per_seq) for x in batch])

    # Create memmap file
    memmap = np.memmap(
        activations_path,
        dtype=save_dtype,
        mode="w+",
        shape=(total_tokens, model.config.hidden_size),
    )

    # Second pass: calculate and save activations
    data_loader.dataset.reset()  # Reset the data loader to start from the beginning
    token_index = 0
    for batch in tqdm(
        data_loader, desc="Calculating activations", total=len(data_loader)
    ):
        activations = get_activations(model, batch)
        batch_tokens = activations.shape[0]
        memmap[token_index : token_index + batch_tokens] = activations.cpu().numpy()
        token_index += batch_tokens

    memmap.flush()

    # Save metadata
    metadata = {
        "total_tokens": total_tokens,
        "d_model": model.config.hidden_size,
        "dtype": save_dtype,
        "model_config": model.config.to_dict(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"Saved activations to {activations_path} and metadata to {metadata_path}")


def custom_str_collate(batch: list[list[str]]) -> list[str]:
    """Collate function to convert batch of lists of strings to a single list of strings"""
    # Flatten the batch of lists into a single list of strings
    return [item for sublist in batch for item in sublist]


class SingleTokenDataset(Dataset):
    def __init__(
        self, filename: str, total_tokens: int, d_model: int, dtype: str = "float32"
    ):
        self.memmap = np.memmap(
            filename, dtype=dtype, mode="r", shape=(total_tokens, d_model)
        )
        self.total_tokens = total_tokens
        self.d_model = d_model

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Have to copy the data because memmap arrays are read-only and
        # PyTorch complains if we try to use them directly
        return torch.as_tensor(self.memmap[idx].copy()).float()


def get_activation_dataset_from_cache(activations_dir: Path) -> SingleTokenDataset:
    """Load a SingleTokenDataset from a directory containing cached activations."""

    activations_dir = Path(activations_dir)
    metadata_path = activations_dir / "metadata.json"
    activations_path = activations_dir / "activations.dat"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    total_tokens = metadata["total_tokens"]
    d_model = metadata["d_model"]

    dataset = SingleTokenDataset(activations_path, total_tokens, d_model)

    return dataset
