from copy import copy
from functools import partial
import torch
from torchvision import transforms

class Shuffle(object):
    def __call__(self, tensor):
        # TODO support gray images
        n_channel, height, width = tensor.size()
        perm = torch.randperm(height*width)
        for c in range(n_channel):
            tensor[c] = tensor[c].view(width*height)[perm].view(height, width)
        return tensor


def make_shuffle_variant(full_dataset):
    d2 = copy(full_dataset)
    d2.ls_transform = transforms.Compose([d2.ls_transform, Shuffle()])
    d2.vs_transform = transforms.Compose([d2.vs_transform, Shuffle()])
    d2.ts_transform = transforms.Compose([d2.ts_transform, Shuffle()])
    return d2


class ShuffleFactory(object):
    @classmethod
    def same_as(cls, full_dataset):
        return make_shuffle_variant(full_dataset)




class InverseFactory(object):
    @classmethod
    def apply_invert(cls, full_dataset, make_copy=True):
        try:
            from PIL.ImageChops import invert
        except ImportError:
            from PIL.ImageOps import invert
        d2 = copy(full_dataset) if make_copy else full_dataset
        d2.ls_transform = transforms.Compose([invert, d2.ls_transform])
        d2.vs_transform = transforms.Compose([invert, d2.vs_transform])
        d2.ts_transform = transforms.Compose([invert, d2.ts_transform])
        return d2


    def __init__(self, full_dataset_factory):
        self.fdf = full_dataset_factory

    def __call__(self, *args, **kwargs):
        full_dataset = self.fdf(*args, **kwargs)
        return self.__class__.apply_invert(full_dataset, make_copy=False)

    def same_as(self, ref_full_dataset):
        fd = self.fdf.same_as(ref_full_dataset)
        return self.__class__.apply_invert(fd, make_copy=False)


# ================================ DATA AUG. ================================= #
class DisablingTransform(object):
    def __init__(self, transform, disabled=False):
        self.transform = transform
        self.disabled = disabled

    def __call__(self, img):
        if self.disabled:
            return img
        return self.transform(img)

    def __repr__(self):
        return "{}({}, disabled={})".format(self.__class__.__name__,
                                            repr(self.transform),
                                            repr(self.disabled))

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False


class TurnOff(object):
    def __init__(self, *disabling_transforms):
        self.disablings = disabling_transforms

    def __enter__(self):
        for disabling in self.disablings:
            disabling.disable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for disabling in self.disablings:
            disabling.enable()



class DataAugmentation(object):
    def get_transform(self):
        return transforms.Compose([])

    def partial(self, full_dataset_factory, **kwargs):
        return partial(full_dataset_factory,
                       ls_data_augmentation=self.get_transform(), **kwargs)


class CropAugmented(DataAugmentation):
    def __init__(self, size=None, crop_size=32, padding=4,
                 padding_mode="reflect"):

        self.size = size
        self.kwargs = {"size": crop_size, "padding":padding,
                       "padding_mode":padding_mode}

    def transform_ls(self):
        ls = []
        if self.size is not None:
            ls.append(transforms.Resize(self.size))

        ls.append(transforms.RandomCrop(**self.kwargs))
        return ls

    def get_transform(self):
        ls = self.transform_ls()
        if len(ls) == 1:
            return ls[0]
        return transforms.Compose(ls)


class CropHzFlipAugmented(CropAugmented):

    def transform_ls(self):
        ls = super().transform_ls()
        ls.append(transforms.RandomHorizontalFlip())
        return ls


class CropHzVFlipAugmented(CropHzFlipAugmented):
    def transform_ls(self):
        ls = super().transform_ls()
        ls.append(transforms.RandomVerticalFlip())
        return ls



class FlipsAugmented(DataAugmentation):
    def get_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
