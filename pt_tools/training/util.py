import warnings

import torch


class Deviceable(object):
    def __init__(self, use_cuda):
        if use_cuda:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                warnings.warn(
                    "Asking for cuda but not available. Falling back on CPU",
                    ResourceWarning)
                self._device = torch.device("cpu")
        else:
            self._device = torch.device("cpu")

    @property
    def device(self):
        return self._device