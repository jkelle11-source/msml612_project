import torch
from torch.utils.data import DataLoader, TensorDataset

def train_loader(dataset, batch_size, generator):

    train_idx = dataset["train_idx"]
    train_data = torch.from_numpy(dataset["values"][train_idx])
    train_mask = torch.from_numpy(dataset["mask"][train_idx])

    # One dataset over both tensors, so a patient's mask is shuffled together with their values.
    return DataLoader(TensorDataset(train_data, train_mask),
                      batch_size = batch_size,
                      shuffle = True,
                      generator = generator)

def to_batch(values, mask, cond_mask):
    n_patients, n_hours = values.shape[0], values.shape[1]
    return {
        "observed_data": values,
        "observed_mask": mask,
        "gt_mask": cond_mask,
        # Hour index 0.0 ... 47.0 for every patient, as eval.EvalSampler gives the model.
        "timepoints": torch.arange(n_hours).float().expand(n_patients, -1)
    }