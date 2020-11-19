from abc import ABCMeta, abstractmethod
from copy import copy

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from torchvision import transforms


def get_data_folder(db_folder):
    import os
    paths = [
        os.path.join(os.environ.get("DATA", "/dev/null"), db_folder),
        os.path.join("./data", db_folder),
        os.path.join(os.path.expanduser("~/data"), db_folder)

    ]

    for path in paths:
        if os.path.exists(path):
            return path

    raise ValueError("No path found for '{}'".format(db_folder))


class FullDataset(object, metaclass=ABCMeta):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Compose([])

    @classmethod
    def get_default_shape(cls):
        return 0, 0, 0

    @classmethod
    def get_default_lengths(cls):
        """
        Return
        ------
        train, val, test: triplet of int
            The sizes of the training, validation and testing sets, respectively
        """
        return 0, 0, 0

    @classmethod
    def get_default_n_outputs(cls):
        return 0

    @classmethod
    def same_as(cls, full_dataset, overriding_shape=None,
                overrinding_normalization=None,
                overrinding_folder_lookup_fn=None):
        shape = full_dataset.shape if overriding_shape is None else overriding_shape
        normalization = full_dataset.normalization if overrinding_normalization is None else overrinding_normalization
        folder_lookup_fn = full_dataset.foler_lookup_fn if overrinding_folder_lookup_fn else overrinding_folder_lookup_fn
        return cls(shape, normalization)

    @classmethod
    def base_transform(cls):
        return list()

    def __init__(self, shape=None, normalization=None,
                 ls_data_augmentation=None, folder_lookup_fn=None):
        def_shape = self.__class__.get_default_shape()
        if shape is None:
            shape = def_shape
        if normalization is None:
            normalization = self.__class__.get_default_normalization()

        # Transform
        # 0. Base transform
        # 1. Data augmentation (PIL)
        # 2. Shape (PIL)
        # 3. Tensor
        # 4. Normalization
        transform_list = self.__class__.base_transform()

        # Shape analysis
        if shape[0] != def_shape[0]:
            transform_list.append(transforms.Grayscale(shape[0]))
        if shape[1] != def_shape[1] or shape[2] != def_shape[2]:
            transform_list.append(transforms.Resize((shape[1], shape[2])))

        transform_list.append(transforms.ToTensor())
        transform_list.append(normalization)

        transform = transforms.Compose(transform_list)
        self.vs_transform = self.ts_transform = transform

        if ls_data_augmentation is not None:
            transform_list = [ls_data_augmentation] + transform_list

        self.ls_transform = transforms.Compose(transform_list)
        # For REPR
        self.shape = shape
        self.normalization = normalization
        self.ls_data_augmentation = ls_data_augmentation
        if folder_lookup_fn is None:
            folder_lookup_fn = get_data_folder
        self.folder_lookup_fn = folder_lookup_fn

    def __repr__(self):
        return "{}(shape={}, normalizaton={}, ls_data_augmentation={})" \
               "".format(self.__class__.__name__,
                         repr(self.shape), repr(self.normalization),
                         repr(self.ls_data_augmentation))

    @property
    def n_outputs(self):
        return self.__class__.get_default_n_outputs()

    @property
    def folder(self):
        return self.folder_lookup_fn(self.__class__.__name__.lower())

    @abstractmethod
    def get_ls_vs_ts(self):
        """
        Return
        ------
        ls, vs, ts: cls:`IndexableDataset`
            The learning, validation and test sets
        """
        return tuple()

    def vs_from_ls(self, train_set, ls_prop=0.9):
        # Validation set from Test set
        valid_set = copy(train_set)
        valid_set.transform = self.vs_transform

        indices = np.arange(len(train_set))
        split = int(len(train_set) * ls_prop)

        train_set = Subset(train_set, indices[:split])
        valid_set = Subset(valid_set, indices[split:])

        return train_set, valid_set

    def to_loaders(self, batch_size, *sets, shuffle=True, num_workers=0,
                   pin_memory=True):
        loaders = []
        for set in sets:
            loaders.append(
                torch.utils.data.DataLoader(set,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory)
            )

        return tuple(loaders)

    def get_loaders(self, ls_batch_size, test_batch_size=1024, num_workers=0,
                    pin_memory=True):
        ls, vs, ts = self.get_ls_vs_ts()
        valid_loader, test_loader = self.to_loaders(test_batch_size, vs, ts,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory)

        (train_loader,) = self.to_loaders(ls_batch_size, ls,
                                          num_workers=num_workers,
                                          pin_memory=pin_memory)

        return train_loader, valid_loader, test_loader


class PartialDataset(object):
    # Factory
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self.kwargs = kwargs

    def get_default_normalization(self):
        return self.factory.get_default_normalization()

    def get_default_shape(self):
        return self.factory.get_default_shape()

    def get_default_lengths(self):
        """
        Return
        ------
        train, val, test: triplet of int
            The sizes of the training, validation and testing sets, respectively
        """
        return self.factory.get_default_lengths()

    def same_as(self, full_dataset):
        return self(full_dataset.shape, full_dataset.normalization)

    def __call__(self, shape=None, normalization=None,
                 ls_data_augmentation=None):
        kwargs = copy(self.kwargs)
        kwargs["shape"] = shape
        kwargs["normalization"] = normalization
        kwargs["ls_data_augmentation"] = ls_data_augmentation
        return self.factory(**kwargs)