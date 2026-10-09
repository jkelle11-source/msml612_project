import yaml
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin',
                'ALT', 'Glucose', 'SaO2', 'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate',
                'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS', 'Cholesterol', 'NISysABP', 'TroponinT',
                'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP', 'Creatinine', 'ALP']

def parse_record(path):
    df = pd.read_csv(path)
    df["Time_Bin"] = df["Time"].str.split(":").str[0].astype(int)
    df_feat_filtered = df[df["Parameter"].isin(FEATURES)]
    df_time_filtered = df_feat_filtered[df_feat_filtered["Time_Bin"] < 48]
    df_dedup = df_time_filtered.drop_duplicates(subset = ["Time_Bin", "Parameter"], keep = "last")
    df_pivot = df_dedup.pivot(index = "Time_Bin", 
                              columns = "Parameter", 
                              values = "Value").reindex(index = range(48), columns = FEATURES)
    return df_pivot.to_numpy()

def parse_raw_data_folder(path):
    paths = sorted(Path(path).glob("*.txt"), key = lambda p: int(p.stem))
    frames = []
    record_ids = []
    for p in paths:
        frames.append(parse_record(p))
        record_ids.append(int(p.stem))
    frames_np = np.stack(frames).astype(np.float32)
    record_ids_np = np.asarray(record_ids)
    return frames_np, record_ids_np
    
def compute_tslo(mask):
    tslo = []
    tslo_t = np.zeros((mask.shape[0], mask.shape[2] ))
    for t in range(mask.shape[1]): 
        sheet = mask[:, t, :]
        tslo_t = np.where(sheet == 1, 0, tslo_t + 1)
        tslo.append(tslo_t)
    return np.stack(tslo, axis = 1).astype(np.float32)

def make_split(n, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = round(0.7 * n)
    n_val = round(0.1 * n)
    return order[:n_train], order[n_train: n_train + n_val], order[n_train + n_val:]

def fit_scaler(values, train_idx):
    train_patients = values[train_idx]
    mean, std = np.nanmean(train_patients, axis = (0, 1)), np.nanstd(train_patients, axis = (0, 1))
    return mean, std

def apply_scaler(values, mean, std):
    return (values - mean) / std

def invert_scaler(scaled, mean, std):
    return scaled * std + mean

def build_dataset(raw_dir, split_seed):
    values, record_ids = parse_raw_data_folder(raw_dir)
    mask = ~np.isnan(values)
    tslo = compute_tslo(mask)
    train_idx, val_idx, test_idx = make_split(len(values), split_seed)
    mean, std = fit_scaler(values, train_idx)
    scaled = apply_scaler(values, mean, std)
    scaled = np.nan_to_num(scaled, nan = 0.0)
    return {
        "delta_obs": tslo,
        "record_ids": record_ids,
        "features": np.array(FEATURES),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "mean": mean,
        "std": std, 
        "values": scaled,
        "mask": mask.astype(np.float32)
    }

def save_dataset(dataset, path):
    np.savez_compressed(path, **dataset)

def load_dataset(path):
    return dict(np.load(path))

if __name__ == "__main__":
    config = yaml.safe_load(open("configs/config.yaml"))
    dataset = build_dataset("data/raw/set-a", config["data"]["split_seed"])
    save_dataset(dataset, "data/processed/physionet_set_a.npz")
        
            