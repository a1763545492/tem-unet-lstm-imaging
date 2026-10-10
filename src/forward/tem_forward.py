import numpy as np
from scipy.special import erf
from typing import Tuple, List, Optional


class TEMForward:
    def __init__(self, time_gates: Optional[np.ndarray] = None):
        self.mu0 = 4 * np.pi * 1e-7
        if time_gates is None:
            self.time_gates = np.logspace(-6, -2, 30)
        else:
            self.time_gates = time_gates

    def calculate_layer_response(self, sigma: float, thickness: float, 
                                 time: np.ndarray) -> np.ndarray:
        tau = self.mu0 * sigma * time / 4.0
        sqrt_tau = np.sqrt(tau)
        erf_val = erf(sqrt_tau)
        response = (2 / np.sqrt(np.pi)) * np.exp(-tau) / sqrt_tau - erf_val
        
        if thickness > 0:
            h = thickness
            k = np.sqrt(self.mu0 * sigma / (4 * time))
            depth_factor = np.exp(-2 * k * h)
            response *= (1 - depth_factor)
        
        return response

    def layered_earth_response(self, conductivities: List[float], 
                               thicknesses: List[float]) -> np.ndarray:
        n_layers = len(conductivities)
        if len(thicknesses) != n_layers:
            raise ValueError("Conductivities and thicknesses must have same length")
        
        response = np.zeros_like(self.time_gates)
        sigma_base = conductivities[-1]
        
        for i in range(n_layers):
            sigma = conductivities[i]
            thickness = thicknesses[i]
            
            if sigma == sigma_base:
                continue
                
            delta_sigma = sigma - sigma_base
            layer_resp = self.calculate_layer_response(sigma, thickness, self.time_gates)
            response += delta_sigma * layer_resp
        
        return np.abs(response)

    def add_noise(self, data: np.ndarray, snr: float = 10.0, 
                  noise_type: str = 'gaussian') -> np.ndarray:
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        elif noise_type == 'impulse':
            noise = np.zeros_like(data)
            impulse_idx = np.random.choice(len(data), int(len(data) * 0.05))
            noise[impulse_idx] = np.random.normal(0, 3 * np.sqrt(noise_power), len(impulse_idx))
        elif noise_type == 'periodic':
            t = np.arange(len(data))
            noise = np.sqrt(noise_power) * np.sin(2 * np.pi * t / 5) * np.random.randn(len(data))
        else:
            noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        
        return data + noise

    def generate_synthetic_data(self, n_samples: int = 1000, 
                                max_layers: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        labels = []
        
        for _ in range(n_samples):
            n_layers = np.random.randint(1, max_layers + 1)
            conductivities = np.random.uniform(0.01, 10, n_layers)
            thicknesses = np.random.uniform(10, 200, n_layers)
            thicknesses[-1] = 0
            
            response = self.layered_earth_response(conductivities.tolist(), thicknesses.tolist())
            snr = np.random.uniform(5, 30)
            noisy_response = self.add_noise(response, snr)
            
            pad_len = max_layers - n_layers
            padded_conductivity = np.pad(conductivities, (0, pad_len), mode='constant')
            padded_thickness = np.pad(thicknesses, (0, pad_len), mode='constant')
            
            data.append(noisy_response)
            labels.append(np.stack([padded_conductivity, padded_thickness]))
        
        return np.array(data), np.array(labels)

    def save_data(self, data: np.ndarray, labels: np.ndarray, file_prefix: str):
        np.save(f'{file_prefix}_data.npy', data)
        np.save(f'{file_prefix}_labels.npy', labels)

    def load_data(self, file_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(f'{file_prefix}_data.npy')
        labels = np.load(f'{file_prefix}_labels.npy')
        return data, labels