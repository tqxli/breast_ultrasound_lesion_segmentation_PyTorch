import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
                              
from preprocessor import BUSIDataProcessor

class BUSIDataLoader(DataLoader):
    def __init__(self, imgs_dir, masks_dir, resize_img, validation_split, batch_size, shuffle, num_workers, pin_memory):
        self.dataset = BUSIDataProcessor(imgs_dir, masks_dir, resize_img=False)
        
        self.n_samples = len(dataset)
        self.shuffle = shuffle
        self.validation_split = validation_split

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        # Shuffle indexes
        np.random.shuffle(idx_full)

        # Validation split can be int (numbers) or percentage
        if isinstance(split, float):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        """
        Choose which specific sampler to use here:
        SequentialSampler: Samples elements sequentially, always in the same order.
        RandomSampler: Samples elements randomly. If without replacement, then sample from a shuffled dataset. 
        SubsetRandomSampler: Samples elements randomly from a given list of indices, without replacement.
        """
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)