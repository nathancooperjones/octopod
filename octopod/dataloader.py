import random

from octopod.vision.helpers import get_number_from_string


class MultiDatasetLoader(object):
    """
    Load datasets for multiple tasks

    Parameters
    ----------
    loader_dict: dict
        dictonary of DataLoaders
    shuffle: Boolean (defaults to True)
        Flag for whether or not to shuffle the data
    """
    def __init__(self, loader_dict, shuffle=True):
        self.loader_dict = loader_dict
        self.shuffle = shuffle

        total_samples = 0
        for key in self.loader_dict.keys():
            total_samples += len(self.loader_dict[key].dataset)

        self.total_samples = total_samples
        self.depth = 9

    def __iter__(self):
        named_batches = []
        iterators = {}

        for key in self.loader_dict.keys():
            if get_number_from_string(key) >= self.depth:
                continue

            current_batches = [key] * len(self.loader_dict[key])
            named_batches += current_batches
            iterators[key] = iter(self.loader_dict[key])

        if self.shuffle:
            random.shuffle(named_batches)

        for key in named_batches:
            yield key, next(iterators[key])

    def __len__(self):
        num_batches = 0

        for key in self.loader_dict.keys():
            if get_number_from_string(key) >= self.depth:
                continue

            num_batches += len(self.loader_dict[key])

        return num_batches

    def set_max_depth(self, depth):
        """Setter method to change maximum depth level."""
        self.depth = depth
