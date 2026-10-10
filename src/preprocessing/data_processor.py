import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict
from scipy.signal import savgol_filter


class DataProcessor:
    def __init__(self, scaler_type: str = 'minmax'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        self.fitted = False

    def clean_data(self, data: np.ndarray, method: str = 'savgol') -> np.ndarray:
        cleaned = []
        for sample in data:
            if method == 'savgol':
                cleaned_sample = savgol_filter(sample, window_length=5, polyorder=2)
            elif method == 'median':
                cleaned_sample = np.median(sample.reshape(-1, 3), axis=1).repeat(3)[:len(sample)]
            elif method == 'moving_avg':
                cleaned_sample = np.convolve(sample, np.ones(3)/3, mode='same')
            else:
                cleaned_sample = sample
            cleaned.append(cleaned_sample)
        return np.array(cleaned)

    def normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        data_shape = data.shape
        data_2d = data.reshape(-1, data_shape[-1])
        
        if fit or not self.fitted:
            self.scaler.fit(data_2d)
            self.fitted = True
        
        normalized = self.scaler.transform(data_2d)
        return normalized.reshape(data_shape)

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        data_shape = data.shape
        data_2d = data.reshape(-1, data_shape[-1])
        denormalized = self.scaler.inverse_transform(data_2d)
        return denormalized.reshape(data_shape)

    def log_transform(self, data: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        return np.log10(data + eps)

    def power_transform(self, data: np.ndarray, power: float = 0.3) -> np.ndarray:
        return np.sign(data) * np.power(np.abs(data), power)

    def prepare_for_unet(self, data: np.ndarray, n_channels: int = 1) -> np.ndarray:
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=1)
        if len(data.shape) == 3:
            if n_channels == 1:
                data = np.repeat(data, n_channels, axis=1)
            else:
                raise ValueError(f"Input data has 1 channel but n_channels={n_channels}")
        return data

    def prepare_labels(self, labels: np.ndarray, max_depth: int = 500) -> np.ndarray:
        conductivity = labels[:, 0, :]
        thickness = labels[:, 1, :]
        
        depth_bins = np.zeros_like(conductivity)
        current_depth = 0
        for i in range(conductivity.shape[1]):
            depth_bins[:, i] = current_depth + thickness[:, i] / 2
            current_depth += thickness[:, i]
        
        conductivity = np.clip(conductivity, 0, None)
        thickness = np.clip(thickness, 0, max_depth)
        
        return np.stack([conductivity, thickness, depth_bins], axis=1)

    def augment_data(self, data: np.ndarray, labels: np.ndarray, 
                     augment_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        augmented_data = [data]
        augmented_labels = [labels]
        
        for _ in range(augment_factor - 1):
            noise = np.random.normal(0, 0.01, data.shape)
            augmented_data.append(data + noise)
            augmented_labels.append(labels)
        
        return np.concatenate(augmented_data), np.concatenate(augmented_labels)

    def split_data(self, data: np.ndarray, labels: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        n_samples = len(data)
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return {
            'train_data': data[train_idx],
            'train_labels': labels[train_idx],
            'val_data': data[val_idx],
            'val_labels': labels[val_idx],
            'test_data': data[test_idx],
            'test_labels': labels[test_idx]
        }

    def save_processor(self, file_path: str):
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_processor(file_path: str) -> 'DataProcessor':
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)