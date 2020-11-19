import numpy as np
import torch
from PIL import Image
from sklearn.utils import check_random_state
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from torchvision import transforms

from .full_dataset import FullDataset, get_data_folder


class Uniform(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.5,), (np.sqrt(1 / 12.),))

    @classmethod
    def get_default_lengths(cls):
        return 36000, 4000, 10000

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    class GeneratedDataset(Dataset):
        def __init__(self, n_instances, shape, n_classes, seed=None,
                     transform=None):
            self.transform = transforms.ToTensor() if transform is None else \
                transform
            rs = check_random_state(seed)
            total_size = tuple([n_instances] + list(shape[1:]) + [shape[0]])

            self.data = (rs.rand(*total_size)*255).astype("uint8")
            self.target = torch.from_numpy(
                rs.randint(0, n_classes, (n_instances,))).float()

        def __getitem__(self, item):
            # To be consistent with other dataset, must return a PIL.Image
            img = self.data[item]
            if img.shape[-1] == 1:
                # PIL does not like having 1 channel
                img = img.squeeze()
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, self.target[item]

        def __len__(self):
            return len(self.data)

    def __init__(self, shape=(3, 32, 32), normalization=None,
                 ls_data_augmentation=None, n_output=10, n_instances=50000,
                 folder_lookup_fn=None):
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation,
                         folder_lookup_fn=folder_lookup_fn)

        self._n_outputs = n_output
        ls_size = int(n_instances * .8)
        ts_size = n_instances - ls_size
        self.train = self.__class__.GeneratedDataset(ls_size, shape, n_output,
                                                     98, self.ls_transform)
        self.test = self.__class__.GeneratedDataset(ts_size, shape, n_output,
                                                    97, self.ts_transform)



    @property
    def n_outputs(self):
        return self._n_outputs


    def get_ls_vs_ts(self):

        train_set, valid_set = self.vs_from_ls(self.train)

        return train_set, valid_set, self.test

class GLikeCif(FullDataset):
    # Like CIFAR 10
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.247, 0.243, 0.261)

    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize(cls.means, cls.stds)

    @classmethod
    def get_default_lengths(cls):
        return 45000, 5000, 10000

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    @classmethod
    def get_default_n_outputs(cls):
        return 10

    class GeneratedDataset(Dataset):
        def __init__(self, n_instances, shape, n_classes, means, stds,
                     seed=None, transform=None):
            self.transform = transforms.ToTensor() if transform is None else \
                transform

            rs = check_random_state(seed)
            total_size = tuple([n_instances] + list(shape[1:]) + [shape[0]])

            data = (rs.normal(means, stds, total_size)) * 255
            self.data = data.clip(0, 255).astype("uint8")
            self.target = torch.from_numpy(
                rs.randint(0, n_classes, (n_instances,))).float()

        def __getitem__(self, item):
            # To be consistent with other dataset, must return a PIL.Image
            img = self.data[item]
            if img.shape[-1] == 1:
                # PIL does not like having 1 channel
                img = img.squeeze()
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, self.target[item]

        def __len__(self):
            return len(self.data)

    def __init__(self, shape=None, normalization=None,
                 ls_data_augmentation=None,
                 folder_lookup_fn=None):
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation,
                         folder_lookup_fn=folder_lookup_fn)

        ls_size, vs_size, ts_size = self.__class__.get_default_lengths()
        shape = self.__class__.get_default_shape()
        n_output = self.__class__.get_default_n_outputs()

        self.train = self.__class__.GeneratedDataset(ls_size, shape, n_output,
                                                     self.__class__.means,
                                                     self.__class__.stds,
                                                     47, self.ls_transform)

        self.val = self.__class__.GeneratedDataset(vs_size, shape, n_output,
                                                   self.__class__.means,
                                                   self.__class__.stds,
                                                   48, self.vs_transform)

        self.test = self.__class__.GeneratedDataset(ts_size, shape, n_output,
                                                    self.__class__.means,
                                                    self.__class__.stds,
                                                    49, self.ts_transform)

    def get_ls_vs_ts(self):
        return self.train, self.val, self.test





class Gaussian(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Compose([])  # Nothing to do

    @classmethod
    def get_default_lengths(cls):
        return 36000, 4000, 10000

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    class GeneratedDataset(Dataset):
        def __init__(self, n_instances, shape, n_classes, sigma, seed=None,
                     transform=None):
            self.transform = transforms.ToTensor() if transform is None else \
                transform

            rs = check_random_state(seed)
            total_size = tuple([n_instances] + list(shape[1:]) + [shape[0]])


            data = 255*(rs.normal(0, sigma, total_size) + 0.5)
            self.data = data.clip(0, 255).astype("uint8")
            self.target = torch.from_numpy(
                rs.randint(0, n_classes, (n_instances,))).float()

        def __getitem__(self, item):
            # To be consistent with other dataset, must return a PIL.Image
            img = self.data[item]
            if img.shape[-1] == 1:
                # PIL does not like having 1 channel
                img = img.squeeze()
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, self.target[item]

        def __len__(self):
            return len(self.data)


    def __init__(self, shape=(3, 32, 32), normalization=None,
                 ls_data_augmentation=None, n_output=10, n_instances=10000,
                 sigma=.25,
                 folder_lookup_fn=None):
        # Note sigma = .25 --> Pr(-.5 < x < .5) = 95%
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation,
                         folder_lookup_fn=folder_lookup_fn)

        self._n_outputs = n_output
        self._sigma = sigma
        ls_size = int(n_instances * .8)
        ts_size = n_instances - ls_size
        self.train = self.__class__.GeneratedDataset(ls_size, shape, n_output,
                                                     self._sigma, 73,
                                                     self.ls_transform)
        self.test = self.__class__.GeneratedDataset(ts_size, shape, n_output,
                                                    self._sigma, 79,
                                                    self.ts_transform)

    @property
    def folder(self):
        return get_data_folder("gaussian")

    @property
    def n_outputs(self):
        return self._n_outputs


    def get_ls_vs_ts(self, ls_transform=None, transform=None):
        train_set, valid_set = self.vs_from_ls(self.train)

        return train_set, valid_set, self.test



class Constant(FullDataset):
    @classmethod
    def get_default_normalization(cls):
        return transforms.Normalize((0.5,), (np.sqrt(1 / 12.),))

    @classmethod
    def get_default_lengths(cls):
        return 5, 2, 3

    @classmethod
    def get_default_shape(cls):
        return 3, 32, 32

    class GeneratedDataset(Dataset):
        def __init__(self, n_instances, shape, n_classes, seed=None,
                     transform=None):
            self.transform = transforms.ToTensor() if transform is None else \
                transform

            values = np.linspace(0, 255, n_instances).astype("uint8")

            rs = check_random_state(seed)
            rs.shuffle(values)

            total_size = tuple([n_instances] + list(shape[1:]) + [shape[0]])
            self.data = np.ones(total_size, dtype="uint8")

            for i, v in enumerate(values):
                self.data[i, ...] *= v

            self.target = torch.from_numpy(
                rs.randint(0, n_classes, (n_instances,))).float()

        def __getitem__(self, item):
            # To be consistent with other dataset, must return a PIL.Image
            img = self.data[item]
            if img.shape[-1] == 1:
                # PIL does not like having 1 channel
                img = img.squeeze()
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, self.target[item]

        def __len__(self):
            return len(self.data)

    def __init__(self, shape=(3, 32, 32), normalization=None,
                 ls_data_augmentation=None, n_output=10, n_instances=10,
                 folder_lookup_fn=None):
        super().__init__(shape=shape, normalization=normalization,
                         ls_data_augmentation=ls_data_augmentation,
                         folder_lookup_fn=folder_lookup_fn)

        self._n_outputs = n_output

        whole_data = self.__class__.GeneratedDataset(n_instances, shape,
                                                     n_output, 103,
                                                     self.ls_transform)

        ls_size = n_instances //2
        ts_size = (n_instances - ls_size) // 2
        vs_size = n_instances - ts_size - ls_size

        s, e = 0, ls_size
        self.train = torch.utils.data.Subset(whole_data, list(range(s, e)))
        s, e = e, e+vs_size
        self.val = torch.utils.data.Subset(whole_data, list(range(s, e)))
        s, e = e, e + ts_size
        self.test = torch.utils.data.Subset(whole_data, list(range(s, e)))

        self.val.transform = self.vs_transform
        self.test.transform = self.ts_transform


    @property
    def folder(self):
        return get_data_folder("uniform")


    @property
    def n_outputs(self):
        return self._n_outputs


    def get_ls_vs_ts(self):
        return self.train, self.val, self.test