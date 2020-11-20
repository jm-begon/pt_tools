from torch.utils.data import DataLoader, ConcatDataset, Subset

def get_base_dataset(dataset):
    if isinstance(dataset, DataLoader):
        return get_base_dataset(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return get_base_dataset(dataset.datasets[0])
    if isinstance(dataset, Subset):
        return get_base_dataset(dataset.dataset)
    return dataset

def get_transform(dataset):
    return get_base_dataset(dataset).transform