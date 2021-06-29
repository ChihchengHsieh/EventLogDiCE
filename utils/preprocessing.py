
from typing import List
from sklearn.model_selection import train_test_split

def dataset_split(data, portions: List[float], seed: int=None, shuffle=True):
    splitted_datasets = []
    assert(sum(portions) < 1)
    left = data
    for i, p in enumerate(portions):
        p_data, left = train_test_split(left, train_size= p/ (1 - sum(portions[:i])), random_state= seed, shuffle=shuffle)
        splitted_datasets.append(p_data)

    splitted_datasets.append(left)

    return splitted_datasets