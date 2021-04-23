import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler
                              
from .preprocessor import BUSIDataProcessor, BUSIDataProcessor_with_labels, TestDataset

class BUSIDataLoader(DataLoader):
    def __init__(self, imgs_dir, masks_dir, resize_img, validation_split, batch_size, shuffle, num_workers, pin_memory, do_classification=False, labels_dir=None):
        if do_classification and labels_dir is not None:
            self.dataset = BUSIDataProcessor_with_labels(imgs_dir, masks_dir, labels_dir, resize_img=True)
            self.normal_samples_idx = self.dataset.get_normal_samples_idx()
        else:
            self.dataset = BUSIDataProcessor(imgs_dir, masks_dir)
            self.normal_samples_idx = []
        
        self.n_samples = len(self.dataset)
        self.shuffle = shuffle
        self.validation_split = validation_split

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': pin_memory
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)
        idx_exclude_from_valid = idx_full[self.normal_samples_idx]
        idx_rest = np.delete(idx_full, idx_exclude_from_valid)

        np.random.seed(0)
        # Shuffle indexes
        np.random.shuffle(idx_rest)

        # Validation split can be int (numbers) or percentage
        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        # Validation only on segmentation
        valid_idx = idx_rest[0:len_valid]
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


# class TestDataLoader(DataLoader):
#     def __init__(self, imgs_dir):
#         self.dataset = TestDataset(imgs_dir)


