import numpy as np
from abc import ABC, abstractmethod

import queue
import threading

def pin_numpy(obj):
    """
    Ensure NumPy arrays are contiguous and own their memory.
    Recursively applies to tuples, lists, dicts.
    """
    if isinstance(obj, np.ndarray):
        if not obj.flags['C_CONTIGUOUS'] or not obj.flags['OWNDATA']:
            return np.ascontiguousarray(obj)
        return obj

    elif isinstance(obj, (tuple, list)):
        return type(obj)(pin_numpy(x) for x in obj)

    elif isinstance(obj, dict):
        return {k: pin_numpy(v) for k, v in obj.items()}

    else:
        return obj
    
class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


class SimpleDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = np.asarray(X)
        self.y = None if y is None else np.asarray(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

class Sampler(ABC):
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

class SequentialSampler:
    def __init__(self, dataset):
        self.n = len(dataset)

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n

class RandomSampler(Sampler):
    def __init__(self, dataset, seed=None):
        self.n = len(dataset)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return iter(self.rng.permutation(self.n))

    def __len__(self):
        return self.n

class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class DataLoader:
    """
    NumPy-based data loader providing PyTorch-compatible batching,
    shuffling, and optional pinned-memory support.

    This class iterates over a dataset and yields mini-batches of data.
    It mirrors the high-level API of ``torch.utils.data.DataLoader``,
    while remaining framework-agnostic and NumPy-first.

    Parameters
    ----------
    dataset : object
        Dataset object implementing ``__len__`` and ``__getitem__``.
        Each ``dataset[i]`` should return a sample or a tuple of samples.

    batch_size : int, default=1
        Number of samples per batch.

    shuffle : bool, default=False
        Whether to shuffle dataset indices at the beginning of each
        iteration.

    num_workers : int, default=0
        Number of worker threads used to prefetch batches.
        If 0, data loading is performed in the main thread.
        Thread-based parallelism is used (not multiprocessing).

    pin_memory : bool, default=False
        If True, batches are copied into pinned host memory.

        - If CuPy is available, NumPy arrays are copied into CUDA
          page-locked (pinned) host memory, enabling fast asynchronous
          host-to-device transfers.
        - If CuPy is not available, arrays are converted to contiguous
          NumPy arrays that own their memory.

        This flag is ignored for non-array objects.

    drop_last : bool, default=False
        If True, drop the last incomplete batch if the dataset size is
        not divisible by ``batch_size``.

    collate_fn : callable, optional
        Function to merge a list of samples into a batch.
        If None, a default collate function is used which stacks NumPy
        arrays along a new leading dimension.

    seed : int, optional
        Random seed used when ``shuffle=True`` to ensure reproducible
        shuffling.

    prefetch : int, default=2
        Maximum number of prefetched batches per worker thread.
        Only relevant when ``num_workers > 0``.

    Notes
    -----
    - This DataLoader does not perform multiprocessing.
    - Memory pinning is implemented using CuPy when available.
    - The loader is lazy and yields batches on demand.
    - Nested batch structures (tuples, lists, dicts) are supported.

    Examples
    --------
    >>> loader = DataLoader(
    ...     dataset,
    ...     batch_size=32,
    ...     shuffle=True,
    ...     num_workers=4,
    ...     pin_memory=True
    ... )
    >>> for x, y in loader:
    ...     pass
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=None,
        seed=None,
        prefetch=2,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn or self.default_collate
        self.seed = seed
        self.prefetch = prefetch

        # --- Sampler logic (PyTorch-like)
        if shuffle:
            sampler = RandomSampler(dataset, seed=seed)
        else:
            sampler = SequentialSampler(dataset)

        self.batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last
        )


    def __iter__(self):
        if self.num_workers <= 0:
            for batch_indices in self.batch_sampler:
                samples = [self.dataset[i] for i in batch_indices]
                out = self.collate_fn(samples)
                if self.pin_memory:
                    out = pin_numpy(out)
                yield out
        else:
            yield from self._threaded_iterator()

    def _threaded_iterator(self):
        q = queue.Queue(maxsize=self.prefetch)
        stop = object()

        def worker():
            for batch_idx in self.batch_sampler:
                batch = [self.dataset[i] for i in batch_idx]
                out = self.collate_fn(batch)
                if self.pin_memory:
                    out = pin_numpy(out)
                q.put(out)
            q.put(stop)

        threads = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(self.num_workers)
        ]

        for t in threads:
            t.start()

        finished = 0
        while finished < self.num_workers:
            item = q.get()
            if item is stop:
                finished += 1
            else:
                yield item

    def __len__(self):
        return len(self.batch_sampler)

    @staticmethod
    def default_collate(samples):
        # Handles tuples or arrays
        if isinstance(samples[0], tuple):
            return tuple(np.stack(items) for items in zip(*samples))
        return np.stack(samples)

