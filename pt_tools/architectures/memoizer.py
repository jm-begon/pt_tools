import torch
import torch.nn as nn
import warnings

class PseudoMemoizer(nn.Module):
    def __init__(self, decorated):
        super().__init__()
        self.decorated = decorated

    def forward(self, x, indices):
        return self.decorated(x)


class Memoizer(nn.Module):
    def __init__(self, decorated):
        super().__init__()
        self.decorated = decorated
        self.memo_table = {}
        self.grad_memo_table = {}

    def forward(self, x, indices=None):
        if indices is None:
            warnings.warn("Using memoizer without indices.")
            return self.decorated(x)

        x_todo = []
        idx = []
        y = []
        # Look up known xi
        for i, (key, xi) in enumerate(zip(indices, x)):
            key = key.item()
            yi = self.memo_table.get(key)
            y.append(yi)
            if yi is None:
                x_todo.append(xi)
                idx.append(i)

        # Compute unknown
        if len(x_todo) > 0:
            x_pred = torch.stack(x_todo, 0)
            indices_pred = torch.LongTensor(idx)
            y_todo = self.decorated(x_pred)

            # Memoize unknown
            for i, yi in zip(idx, y_todo):
                key = indices[i].item()
                self.memo_table[key] = y[i] = yi

        return torch.stack(y, 0)

    def get_grad(self, indices, outputs, inputs, grad_outputs):
        x_todo = []
        idx = []
        g = []
        # Look up known xi
        for i, (key, xi, yi) in enumerate(zip(indices, inputs, outputs)):
            key = key.item()
            yi = self.memo_table.get(key)
            g.append(yi)
            if yi is None:
                x_todo.append(xi)
                idx.append(i)

        # Compute unknown
        if len(x_todo) > 0:
            x_pred = torch.stack(x_todo, 0)
            indices_pred = torch.LongTensor(idx)
            y_todo = self.decorated(x_pred, indices_pred)

            # Memoize unknown
            for i, yi in zip(idx, y_todo):
                key = indices[i].item()
                self.memo_table[key] = g[i] = yi

        return torch.stack(g, 0)


class SlowMemoizer(nn.Module):
    def __init__(self, decorated):
        super().__init__()
        self.decorated = decorated
        self.memo_table = {}

    def hash(self, xi):
        key = "{}_{}_{}_{}".format(xi, xi.min(), xi.mean(), xi.max())
        return key

    def forward(self, x):
        x_todo = []
        idx = []
        y = []
        keys = []
        # Look up known xi
        for i, xi in enumerate(x):
            key = self.hash(xi)
            yi = self.memo_table.get(key)
            if yi is None:
                x_todo.append(xi)
                idx.append(i)
                y.append(None)
                keys.append(key)
            else:
                y.append(yi)

        # Compute unknown
        if len(x_todo) > 0:
            x_pred = torch.stack(x_todo, 0)
            y_todo = self.decorated(x_pred)

            # Memoize unknown
            for i, key, yi in zip(idx, keys, y_todo):
                self.memo_table[key] = yi
                y[i] = yi

        return torch.stack(y, 0)


def test_memoizer(num_workers=0, use_cuda=False):
    import time
    from .arch_32x32 import ResNet50
    from ..datasets import CIFAR10, CIFAR100, SVHN
    from ..training import Tester
    for db in CIFAR10, CIFAR100, SVHN:
        print("Test memoizer for", db)
        dataset = db(42, num_workers=num_workers)
        loader, _, _ = dataset.get_loaders(128)

        # Raw
        model = ResNet50(n_outputs=dataset.n_outputs)
        # tester = Tester(model, use_cuda=use_cuda)
        # start = time.time()
        # print("\tRaw model:", tester.test(loader), "(in {})".format(time.time() - start))

        # Memo
        model = Memoizer(model)
        tester = Tester(model, use_cuda=use_cuda)

        start = time.time()
        print("\tMemo first pass:", tester.test(loader), "(in {})".format(time.time()-start))
        print("\t\t#Points", len(model.memo_table))

        start = time.time()
        print("\tMemo second pass:", tester.test(loader), "(in {})".format(time.time() - start))
        print("\t\t#Points", len(model.memo_table))