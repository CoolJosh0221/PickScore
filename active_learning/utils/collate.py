"""
Collate function for handling different data types in PyTorch DataLoaders.
Optimized for GPU usage and efficient batching.
"""

from typing import List, Dict, Any


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Custom collate function to handle PIL images and other data types efficiently.

    Args:
        batch: List of dictionaries containing sample data

    Returns:
        Dictionary with batched data, preserving original data types
    """
    if not batch:
        return {}

    keys = batch[0].keys()
    collated_batch = {k: [d[k] for d in batch] for k in keys}

    return collated_batch
