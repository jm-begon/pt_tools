"""
Class |  STL 10  |  CIFAR 10
------+----------+------------
  0   | airplane | airplane
  1   | bird     | automobile
  2   | car      | bird
  3   | cat      | cat
  4   | deer     | deer
  5   | dog      | dog
  6   | horse    | frog
  7   | monkey   | horse
  8   | ship     | ship
  9   | truck    | truck
Need to switch classes 1 and 2 and classes 6 and 7.
Note that monkey (STL 10) is not the same as frog (CIFAR 10)
"""
import os
from abc import ABCMeta

import numpy as np
import torchvision
from torchvision import transforms

from .full_dataset import FullDataset, get_data_folder





class CIFAR10(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.247, 0.243, 0.261))

    @classmethod
    def get_default_lengths(cls):
        return 45000, 5000, 10000

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    @classmethod
    def get_default_n_outputs(cls):
        return 10


    def _get_pytorch_class(self):
        return torchvision.datasets.CIFAR10

    def get_ls_vs_ts(self):
        train_set = self._get_pytorch_class()(
            root=self.folder, train=True,
            download=True,
            transform=self.ls_transform,
        )

        test_set = self._get_pytorch_class()(
            root=self.folder, train=False,
            download=True,
            transform=self.ts_transform,
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set



class CIFAR100(CIFAR10):

    @classmethod
    def get_default_n_outputs(cls):
        return 100

    def _get_pytorch_class(self):
        return torchvision.datasets.CIFAR100


class SVHN(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    @classmethod
    def get_default_lengths(cls):
        return 65931, 7326, 26032

    @classmethod
    def get_default_n_outputs(cls):
        return 10

    def get_ls_vs_ts(self):
        train_set = torchvision.datasets.SVHN(
            root=self.folder, split="train",
            download=True,
            transform=self.ls_transform,
        )

        test_set = torchvision.datasets.SVHN(
            root=self.folder, split="test",
            download=True,
            transform=self.ts_transform,
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set

class MNISTLike(FullDataset):
    @classmethod
    def get_default_lengths(cls):
        return 54000, 6000, 10000

    @classmethod
    def get_default_shape(cls):
        return 1, 28, 28

    @classmethod
    def get_default_n_outputs(cls):
        return 10

    @classmethod
    def get_default_normalization(cls):
        return AttributeError("Abstract")

    @classmethod
    def torch_class(cls):
        return AttributeError("Abstract")

    def get_ls_vs_ts(self):
        train_set = self.__class__.torch_class()(
            root=self.folder, train=True,
            download=True,
            transform=self.ls_transform
        )

        test_set = self.__class__.torch_class()(
            root=self.folder, train=False,
            download=True,
            transform=self.ts_transform
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set

class MNIST(MNISTLike):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.1307,), (0.3081,))

    @classmethod
    def torch_class(cls):
        return torchvision.datasets.MNIST


class FashionMNIST(MNISTLike):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.2860,), (0.3530,))

    @classmethod
    def torch_class(cls):
        return torchvision.datasets.FashionMNIST



class KMNIST(MNIST):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.19164627944577953,), (0.3482464146758613,))

    @classmethod
    def get_default_normalization(cls):
        return transforms.Compose([])

    @classmethod
    def torch_class(cls):
        return torchvision.datasets.KMNIST


class STL10(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.4311,), (0.2634,))

    @classmethod
    def get_default_lengths(cls):
        return 4500, 500, 8000

    @classmethod
    def get_default_shape(cls):
        return 3, 93, 93

    @classmethod
    def get_default_n_outputs(cls):
        return 10


    def get_ls_vs_ts(self):
        train_set = torchvision.datasets.STL10(
            root=self.folder, split='train',
            download=True,
            transform=self.ls_transform,
        )

        test_set = torchvision.datasets.STL10(
            root=self.folder, split='test',
            download=True,
            transform=self.ts_transform
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set

    def get_monkeys(self):
        ls, vs, ts = self.get_ls_vs_ts()
        all = []
        for samples in ls, ts:
            dataset = samples
            while(hasattr(dataset, "dataset")):
                # In case of concat. dataset
                dataset = dataset.dataset
            monkeys = dataset.labels == 7
            all.append(monkeys)
        return np.concatenate(all)


class STL10WithUnlabeled(STL10):
    @classmethod
    def get_default_lengths(cls):
        return 94500, 10500, 8000

    @classmethod
    def get_default_n_outputs(cls):
        return 11


    def get_ls_vs_ts(self):
        train_set = torchvision.datasets.STL10(
            root=self.folder, split='train+unlabeled',
            download=True,
            transform=self.ls_transform,
            target_transform=(lambda t: 10 if t is None else t),
        )

        test_set = torchvision.datasets.STL10(
            root=self.folder, split='test',
            download=True,
            transform=self.ts_transform
        )

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set

    @property
    def folder(self):
        return super().folder  # look for 'stl10'

    @property
    def folder(self):
        return self.folder_lookup_fn("stl10")




class TinyImageNet(FullDataset):
    """
    From https://tiny-imagenet.herokuapp.com/

    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    """
    @classmethod
    def get_default_normalization(cls):
        # Computed on training set
        return transforms.Normalize((0.4802458005453784, 0.44807219498302625,
                                     0.3975477610692504),
                                    (0.2769864106388343, 0.2690644893481639,
                                     0.2820819105768366))

    @classmethod
    def get_default_lengths(cls):
        return 72000, 8000, 20000

    @classmethod
    def get_default_shape(cls):
        return 3, 64, 64

    @classmethod
    def get_default_n_outputs(cls):
        return 200   # 500 of each

    @property
    def folder(self):
        return self.folder_lookup_fn("tinyimagenet/tiny-imagenet-200")

    @property
    def _shuffle(self):
        return True

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            self.normalization
        ])

    def _get_full_set(self, subset, transform):
        path = os.path.join(self.folder, subset)
        return torchvision.datasets.ImageFolder(root=path,
                                                transform=transform)

    def get_ls_vs_ts(self):
        train_set = self._get_full_set("train", self.ls_transform)
        test_set = self._get_full_set("test", self.ts_transform)

        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set


class LSUNTestSet(FullDataset):
    """
    From https://github.com/fyu/lsun

    conda install lmdb
    """
    @classmethod
    def get_default_normalization(cls):
        return transforms.Compose([])  # TODO

    @classmethod
    def get_default_shape(cls):
        return 3, 256, 256

    def __init__(self, shape=None, normalization=None,
                 ls_data_augmentation=None):
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation)
        if shape is None:
            self.ls_transform = transforms.Compose([transforms.Resize(256, 256),
                                                    self.ls_transform])
            self.vs_transform = transforms.Compose([transforms.Resize(256, 256),
                                                    self.vs_transform])
            self.ts_transform = transforms.Compose([transforms.Resize(256, 256),
                                                    self.ts_transform])

    @classmethod
    def get_default_n_outputs(cls):
        return 0

    @property
    def n_outputs(self):
        raise NotImplementedError()

    @property
    def folder(self):
        return get_data_folder("lsun")

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            self.normalization
        ])

    def _get_full_set(self, transform):
        return torchvision.datasets.LSUN(root=self.folder,
                                         classes="test",
                                         transform=transform)

    def get_ls_vs_ts(self):
        actual_test_set = self._get_full_set(self.ts_transform)
        # TODO beware of transformation
        train_set, test_set = self.vs_from_ls(actual_test_set, 0.8)
        train_set, valid_set = self.vs_from_ls(train_set)

        return train_set, valid_set, test_set


class TrValTsDataset(FullDataset, metaclass=ABCMeta):
    def get_ls_vs_ts(self):
        ls = torchvision.datasets.ImageFolder(
            root=os.path.join(self.folder, "train"),
            transform=self.ls_transform
        )
        vs = torchvision.datasets.ImageFolder(
            root=os.path.join(self.folder, "val"),
            transform=self.vs_transform
        )
        ts = torchvision.datasets.ImageFolder(
            root=os.path.join(self.folder, "test"),
            transform=self.ts_transform
        )
        return ls, vs, ts



class ImageNet(FullDataset):
    """
    See http://www.image-net.org/challenges/LSVRC/2012/

        - Download from http://image-net.org/challenges/LSVRC/2012/downloads.php#images
        - Use the Pytorch class to ready everithing

    https://stackoverflow.com/questions/40744700/how-can-i-find-imagenet-data-labels

    """

    @classmethod
    def get_default_normalization(cls):
        # stats from https://pytorch.org/docs/stable/torchvision/models.html
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        # stats estimated on training set
        # return transforms.Normalize(
        #     (0.48026870993652526, 0.45750730332850736, 0.4081817554661841),
        #     (0.2807399958925809, 0.27367912125650207, 0.28782503124759895))

    @classmethod
    def get_default_n_outputs(cls):
        return 1000  # ~ [1000, 1300] / cls

    @classmethod
    def get_default_lengths(cls):
        return 1281167, 50000, 100000

    @classmethod
    def get_default_shape(cls):
        return 3, 224, 224

    @classmethod
    def base_transform(cls):
        return [transforms.Resize(256), transforms.CenterCrop(224)]

    @property
    def _shuffle(self):
        return True

    def get_ls_vs_ts(self):
        ls = torchvision.datasets.ImageNet(
            root=self.folder,
            split="train",
            transform=self.ls_transform
        )

        ts = torchvision.datasets.ImageNet(
            root=self.folder,
            split="val",
            transform=self.ts_transform
        )

        ls, vs = self.vs_from_ls(ls, 0.9)

        # ts = torchvision.datasets.ImageFolder(
        #     root=os.path.join(self.folder, "alltest"),
        #     transform=self.ts_transform,
        #     target_transform=lambda x:-1  # unknown
        # )
        return ls, vs, ts


