from ilit.utils.utility import LazyImport
from abc import abstractmethod
import collections
import numpy as np
from .sampler import IterableSampler, SequentialSampler, BatchSampler
from .base_dataloader import BaseDataLoader

tf = LazyImport('tensorflow')

class TensorflowDataLoader(BaseDataLoader):
    """DataLoader for frameework Tensorflow, if it's a tf.data.Dataset we will directly use
       the dataloader in the other case will use DefaultDataLoader instead.

    """

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
        sampler, batch_sampler, num_workers, pin_memory):

        drop_last = False if last_batch == 'rollover' else True

        if isinstance(dataset, tf.data.Dataset):
            return dataset.batch(batch_size, drop_remainder=drop_last)
        else:
            return DefaultDataLoader(dataset, batch_size, collate_fn,
                sampler, batch_sampler, drop_last, num_workers, pin_memory)

def default_collate(batch):
    """Puts each data field into a pd frame with outer dimension batch size"""
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [default_collate(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        return np.stack(batch)
    else:
        return batch

class Fetcher(object):
    def __init__(self, dataset, collate_fn, drop_last):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    @abstractmethod
    def __call__(self, batched_indices):
        raise NotImplementedError


class IterableFetcher(Fetcher):
    def __init__(self, dataset, collate_fn, drop_last):
        super(IterableFetcher, self).__init__(dataset, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)

    def __call__(self, batched_indices):
        data = []
        for _ in batched_indices:
            try:
                data.append(next(self.dataset_iter))
            except StopIteration:
                break
        if len(data) == 0 or (self.drop_last and len(data) < len(batched_indices)):
            raise StopIteration
        return self.collate_fn(data)

class IndexFetcher(Fetcher):
    def __init__(self, dataset, collate_fn, drop_last):
        super(IndexFetcher, self).__init__(dataset, collate_fn, drop_last)

    def __call__(self, batched_indices):
        data = [self.dataset[idx] for idx in batched_indices]

        return self.collate_fn(data)

FETCHERS = {"index":IndexFetcher, "iter":IterableFetcher, }

class DefaultDataLoader(BaseDataLoader):
    """In tensorflow1.x dataloader is coupled with the graph, but it also support feed_dict
       method to do session run, this dataloader is designed to satisfy the usage of feed dict
       in tf1.x. Although it's a general dataloader and can be used in MXNet and PyTorch.

    """

    def __init__(self, dataset, batch_size=1, last_batch='rollover', collate_fn=None,
        sampler=None, batch_sampler=None, num_workers=0, pin_memory=False):

        self.dataset = dataset
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        if self.collate_fn == None:
            self.collate_fn = default_collate
        
        self._generate_dataloader(self.dataset, batch_size=batch_size,
            last_batch=last_batch, collate_fn=self.collate_fn, sampler=sampler,
            batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)

    def batch(self, batch_size, last_batch='rollover'):
        self._generate_dataloader(self.dataset, batch_size,
            last_batch, self.collate_fn, self.sampler, self.batch_sampler,
            self.num_workers, self.pin_memory)

    @property
    def dataloader(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        batched_indices = next(self.sampler_iter)
        data = self.fetcher(batched_indices)
        return data

    def _generate_sampler(self, dataset):
        if hasattr(dataset, "__getitem__"):
            self.dataset_type = 'index'
            return SequentialSampler(self.dataset)
        elif hasattr(dataset, "__iter__"):
            self.dataset_type = 'iter'
            return IterableSampler()
        else:
            raise ValueError("dataset type only support (index, iter)")

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn,
        sampler, batch_sampler, num_workers, pin_memory):

        drop_last = False if last_batch == 'rollover' else True
        sampler = self._generate_sampler(dataset)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.sampler_iter = iter(self.batch_sampler)
        self.fetcher = FETCHERS[self.dataset_type](dataset, collate_fn, drop_last)
        
