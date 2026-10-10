import numpy as np
import os
from typing import Tuple, List, Dict
from src.forward.tem_forward import TEMForward


class AutoLabelingPipeline:
    def __init__(self, forward_model: TEMForward = None):
        if forward_model is None:
            self.forward_model = TEMForward()
        else:
            self.forward_model = forward_model

    def generate_labels_from_params(self, params: List[Dict]) -> np.ndarray:
        labels = []
        for param in params:
            conductivities = param.get('conductivities', [1.0])
            thicknesses = param.get('thicknesses', [100])
            
            pad_len = param.get('max_layers', 5) - len(conductivities)
            padded_conductivity = np.pad(np.array(conductivities), (0, pad_len), mode='constant')
            padded_thickness = np.pad(np.array(thicknesses), (0, pad_len), mode='constant')
            
            labels.append(np.stack([padded_conductivity, padded_thickness]))
        
        return np.array(labels)

    def generate_data_label_pair(self, n_samples: int = 1000, 
                                 max_layers: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        data = []
        labels = []
        params = []
        
        for _ in range(n_samples):
            n_layers = np.random.randint(1, max_layers + 1)
            conductivities = np.random.uniform(0.01, 10, n_layers).tolist()
            thicknesses = np.random.uniform(10, 200, n_layers).tolist()
            thicknesses[-1] = 0
            
            response = self.forward_model.layered_earth_response(conductivities, thicknesses)
            snr = np.random.uniform(5, 30)
            noisy_response = self.forward_model.add_noise(response, snr)
            
            pad_len = max_layers - n_layers
            padded_conductivity = np.pad(np.array(conductivities), (0, pad_len), mode='constant')
            padded_thickness = np.pad(np.array(thicknesses), (0, pad_len), mode='constant')
            
            data.append(noisy_response)
            labels.append(np.stack([padded_conductivity, padded_thickness]))
            params.append({
                'conductivities': conductivities,
                'thicknesses': thicknesses,
                'snr': snr,
                'n_layers': n_layers
            })
        
        return np.array(data), np.array(labels), params

    def generate_with_terrain(self, terrain_model, n_samples: int = 1000, 
                              max_layers: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        data = []
        labels = []
        params = []
        
        for _ in range(n_samples):
            n_layers = np.random.randint(1, max_layers + 1)
            conductivities = np.random.uniform(0.01, 10, n_layers).tolist()
            thicknesses = np.random.uniform(10, 200, n_layers).tolist()
            thicknesses[-1] = 0
            
            n_stations = np.random.randint(5, 20)
            stations = np.random.uniform(-100, 100, (n_stations, 2))
            
            responses = terrain_model.calculate_terrain_response(
                stations, conductivities, thicknesses, self.forward_model
            )
            
            snr = np.random.uniform(5, 30)
            noisy_responses = []
            for resp in responses:
                noisy_responses.append(self.forward_model.add_noise(resp, snr))
            
            pad_len = max_layers - n_layers
            padded_conductivity = np.pad(np.array(conductivities), (0, pad_len), mode='constant')
            padded_thickness = np.pad(np.array(thicknesses), (0, pad_len), mode='constant')
            
            data.append(np.array(noisy_responses))
            labels.append(np.stack([padded_conductivity, padded_thickness]))
            params.append({
                'conductivities': conductivities,
                'thicknesses': thicknesses,
                'snr': snr,
                'n_layers': n_layers,
                'n_stations': n_stations,
                'stations': stations.tolist()
            })
        
        return np.array(data), np.array(labels), params

    def save_dataset(self, data: np.ndarray, labels: np.ndarray, params: List[Dict], 
                     save_dir: str = './data', dataset_name: str = 'dataset'):
        os.makedirs(save_dir, exist_ok=True)
        
        data_path = os.path.join(save_dir, f'{dataset_name}_data.npy')
        labels_path = os.path.join(save_dir, f'{dataset_name}_labels.npy')
        params_path = os.path.join(save_dir, f'{dataset_name}_params.npy')
        
        np.save(data_path, data)
        np.save(labels_path, labels)
        np.save(params_path, np.array(params, dtype=object))
        
        print(f"Dataset saved to {save_dir}")
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")

    def load_dataset(self, save_dir: str = './data', dataset_name: str = 'dataset') -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        data_path = os.path.join(save_dir, f'{dataset_name}_data.npy')
        labels_path = os.path.join(save_dir, f'{dataset_name}_labels.npy')
        params_path = os.path.join(save_dir, f'{dataset_name}_params.npy')
        
        data = np.load(data_path)
        labels = np.load(labels_path)
        params = np.load(params_path, allow_pickle=True).tolist()
        
        return data, labels, params

    def generate_batch(self, batch_size: int = 32, max_layers: int = 5, 
                       with_terrain: bool = False, terrain_model = None) -> Tuple[np.ndarray, np.ndarray]:
        if with_terrain and terrain_model is not None:
            data, labels, _ = self.generate_with_terrain(terrain_model, batch_size, max_layers)
        else:
            data, labels, _ = self.generate_data_label_pair(batch_size, max_layers)
        
        return data, labels

    def create_training_pipeline(self, config: Dict) -> Tuple[np.ndarray, np.ndarray, 
                                                              np.ndarray, np.ndarray]:
        n_train = config.get('n_train', 10000)
        n_val = config.get('n_val', 2000)
        max_layers = config.get('max_layers', 5)
        with_terrain = config.get('with_terrain', False)
        terrain_model = config.get('terrain_model', None)
        
        if with_terrain and terrain_model is not None:
            train_data, train_labels, _ = self.generate_with_terrain(terrain_model, n_train, max_layers)
            val_data, val_labels, _ = self.generate_with_terrain(terrain_model, n_val, max_layers)
        else:
            train_data, train_labels, _ = self.generate_data_label_pair(n_train, max_layers)
            val_data, val_labels, _ = self.generate_data_label_pair(n_val, max_layers)
        
        return train_data, train_labels, val_data, val_labels