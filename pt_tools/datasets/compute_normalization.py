from collections import defaultdict

import torch
from pt_inspector.stat import Stat
from samplefree_ood.datasets.datasets import GLikeCif
from torchvision import transforms

if __name__ == '__main__':
    collector = {}

    what = "ls"

    full_dataset = GLikeCif(normalization=transforms.Compose([]))
    ls, vs, ts = full_dataset.get_ls_vs_ts()

    if what == "ls":
        dataset = ls
    elif what == "vs":
        dataset = vs
    elif what == "ts":
        dataset = ts
    else:
        dataset = torch.utils.data.ConcatDataset([ls, vs, ts])


    # dataset[0] is a pair (image tensor, label)
    shape = dataset[0][0].size()
    if len(shape) == 2:
        # 1 channel
        n_channels = 1
        get_channel = (lambda x, _: x)
    else:
        n_channels = shape[0]
        get_channel = (lambda x, i: x[i])

    stats = [Stat() for _ in range(n_channels)]
    hist_shape = defaultdict(int)
    hist_cls = defaultdict(int)

    n = 0
    for n, t in enumerate(dataset):
        x = t[0].numpy()
        y = t[1]
        if not isinstance(y, int):
            try:
                y = t[1].item()
            except Exception:
                y = int(t[1])

        hist_cls[y] += 1
        hist_shape[x.shape] += 1
        for c in range(n_channels):
            stats[c].add(get_channel(x, c))

    collector["n_samples"] = n + 1
    collector["hist_shape"] = hist_shape
    collector["hist_cls"] = hist_cls

    means = []
    stds = []
    for stat in stats:
        m, s = stat.get_running()
        means.append(m)
        stds.append(s)


    collector["means"] = tuple(means)
    collector["stds"] = tuple(stds)