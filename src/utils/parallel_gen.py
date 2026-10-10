import numpy as np
import multiprocessing as mp
from typing import Tuple, List, Callable
from functools import partial


def _generate_chunk_helper(args: Tuple[int, Callable, dict]) -> Tuple[np.ndarray, np.ndarray]:
    chunk_size, generator_func, kwargs = args
    data = []
    labels = []
    
    for _ in range(chunk_size):
        sample_data, sample_label = generator_func(**kwargs)
        data.append(sample_data)
        labels.append(sample_label)
    
    return np.array(data), np.array(labels)


class ParallelGenerator:
    def __init__(self, num_workers: int = None):
        if num_workers is None:
            self.num_workers = mp.cpu_count()
        else:
            self.num_workers = min(num_workers, mp.cpu_count())

    def generate_parallel(self, generator_func: Callable, n_samples: int, 
                          **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        chunk_size = n_samples // self.num_workers
        chunks = []
        
        for i in range(self.num_workers):
            if i == self.num_workers - 1:
                chunks.append((n_samples - i * chunk_size, generator_func, kwargs))
            else:
                chunks.append((chunk_size, generator_func, kwargs))
        
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.num_workers) as pool:
            results = pool.map(_generate_chunk_helper, chunks)
        
        data = np.concatenate([r[0] for r in results], axis=0)
        labels = np.concatenate([r[1] for r in results], axis=0)
        
        return data, labels

    def generate_with_terrain(self, forward_model, terrain_model, 
                              n_samples: int, max_layers: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        def generate_sample():
            n_layers = np.random.randint(1, max_layers + 1)
            conductivities = np.random.uniform(0.01, 10, n_layers).tolist()
            thicknesses = np.random.uniform(10, 200, n_layers).tolist()
            thicknesses[-1] = 0
            
            n_stations = np.random.randint(5, 20)
            stations = np.random.uniform(-100, 100, (n_stations, 2))
            
            responses = terrain_model.calculate_terrain_response(
                stations, conductivities, thicknesses, forward_model
            )
            
            snr = np.random.uniform(5, 30)
            noisy_responses = []
            for resp in responses:
                noisy_responses.append(forward_model.add_noise(resp, snr))
            
            pad_len = max_layers - n_layers
            padded_conductivity = np.pad(np.array(conductivities), (0, pad_len), mode='constant')
            padded_thickness = np.pad(np.array(thicknesses), (0, pad_len), mode='constant')
            
            return np.array(noisy_responses), np.stack([padded_conductivity, padded_thickness])
        
        return self.generate_parallel(generate_sample, n_samples)

    def process_parallel(self, data: np.ndarray, process_func: Callable, 
                         **kwargs) -> np.ndarray:
        chunk_size = len(data) // self.num_workers
        chunks = []
        
        for i in range(self.num_workers):
            if i == self.num_workers - 1:
                chunks.append((data[i * chunk_size:], process_func, kwargs))
            else:
                chunks.append((data[i * chunk_size:(i + 1) * chunk_size], process_func, kwargs))
        
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_chunk, chunks)
        
        return np.concatenate(results, axis=0)

    def _process_chunk(self, args: Tuple[np.ndarray, Callable, dict]) -> np.ndarray:
        chunk, process_func, kwargs = args
        results = []
        for sample in chunk:
            results.append(process_func(sample, **kwargs))
        return np.array(results)

    def benchmark(self, generator_func: Callable, n_samples: int, **kwargs):
        import time
        
        start_time = time.time()
        data_single, labels_single = [], []
        for _ in range(n_samples):
            d, l = generator_func(**kwargs)
            data_single.append(d)
            labels_single.append(l)
        single_time = time.time() - start_time
        
        start_time = time.time()
        data_multi, labels_multi = self.generate_parallel(generator_func, n_samples, **kwargs)
        multi_time = time.time() - start_time
        
        speedup = single_time / multi_time
        print(f"单进程耗时: {single_time:.2f}s")
        print(f"多进程耗时: {multi_time:.2f}s")
        print(f"加速比: {speedup:.2f}x")
        
        return speedup