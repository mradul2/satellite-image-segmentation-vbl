import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def datasetSplitter(dataset, validation_split, shuffle):
	random_seed= 42
	dataset_size = len(dataset)

	indices = list(range(dataset_size))
	split = int(np.floor(validation_split * dataset_size))

	if shuffle:
	    np.random.seed(random_seed)
	    np.random.shuffle(indices)
        
	train_indices, val_indices = indices[split:], indices[:split]

	# Creating PT data samplers and loaders:
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)

	return train_sampler, valid_sampler