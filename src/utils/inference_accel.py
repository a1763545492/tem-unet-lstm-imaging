import torch
import torch.nn as nn
import time
from typing import Optional


class InferenceAccelerator:
    def __init__(self, model: nn.Module, device: str = 'auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        
        self.jit_model = None
        self.trt_model = None
        self.accelerator_type = 'none'

    def optimize_with_jit(self, example_input: torch.Tensor):
        example_input = example_input.to(self.device)
        self.jit_model = torch.jit.trace(self.model, example_input)
        self.accelerator_type = 'jit'
        print("Model optimized with Torch JIT")

    def optimize_with_trt(self, example_input: torch.Tensor, 
                          fp16: bool = True, 
                          max_batch_size: int = 32):
        try:
            from torch_tensorrt import compile
            
            input_spec = [torch.TensorRT.Input(
                example_input.shape,
                dtype=torch.float32,
                format=torch.channel_last
            )]
            
            self.trt_model = compile(
                self.model,
                inputs=input_spec,
                enabled_precisions={torch.float16} if fp16 else {torch.float32},
                max_batch_size=max_batch_size
            )
            self.accelerator_type = 'tensorrt'
            print("Model optimized with TensorRT")
        except ImportError:
            print("TensorRT not available, falling back to JIT")
            self.optimize_with_jit(example_input)
        except Exception as e:
            print(f"TensorRT optimization failed: {e}, falling back to JIT")
            self.optimize_with_jit(example_input)

    def infer(self, input_data: torch.Tensor) -> torch.Tensor:
        input_data = input_data.to(self.device)
        
        with torch.no_grad():
            if self.accelerator_type == 'tensorrt' and self.trt_model is not None:
                output = self.trt_model(input_data)
            elif self.accelerator_type == 'jit' and self.jit_model is not None:
                output = self.jit_model(input_data)
            else:
                output = self.model(input_data)
        
        if isinstance(output, tuple):
            output = output[0]
        
        return output.cpu()

    def benchmark(self, input_data: torch.Tensor, n_runs: int = 100) -> dict:
        input_data = input_data.to(self.device)
        results = {}
        
        with torch.no_grad():
            start = time.time()
            for _ in range(n_runs):
                _ = self.model(input_data)
            original_time = (time.time() - start) / n_runs
            results['original'] = original_time
        
        if self.accelerator_type == 'jit' and self.jit_model is not None:
            with torch.no_grad():
                start = time.time()
                for _ in range(n_runs):
                    _ = self.jit_model(input_data)
                jit_time = (time.time() - start) / n_runs
            results['jit'] = jit_time
            results['jit_speedup'] = original_time / jit_time
        
        if self.accelerator_type == 'tensorrt' and self.trt_model is not None:
            with torch.no_grad():
                start = time.time()
                for _ in range(n_runs):
                    _ = self.trt_model(input_data)
                trt_time = (time.time() - start) / n_runs
            results['tensorrt'] = trt_time
            results['tensorrt_speedup'] = original_time / trt_time
        
        print(f"Original: {original_time*1000:.2f} ms")
        if 'jit' in results:
            print(f"JIT: {jit_time*1000:.2f} ms (Speedup: {results['jit_speedup']:.2f}x)")
        if 'tensorrt' in results:
            print(f"TensorRT: {trt_time*1000:.2f} ms (Speedup: {results['tensorrt_speedup']:.2f}x)")
        
        return results

    def save_optimized_model(self, path: str):
        if self.accelerator_type == 'jit' and self.jit_model is not None:
            self.jit_model.save(path)
        elif self.accelerator_type == 'tensorrt' and self.trt_model is not None:
            torch.save(self.trt_model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_optimized_model(self, path: str):
        try:
            self.jit_model = torch.jit.load(path)
            self.accelerator_type = 'jit'
            print("Loaded JIT optimized model")
        except:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.accelerator_type = 'none'
            print("Loaded standard model")


class BatchInferenceEngine:
    def __init__(self, model: nn.Module, batch_size: int = 32):
        self.accelerator = InferenceAccelerator(model)
        self.batch_size = batch_size

    def process_batch(self, data: torch.Tensor) -> torch.Tensor:
        n_samples = data.size(0)
        outputs = []
        
        for i in range(0, n_samples, self.batch_size):
            batch = data[i:i+self.batch_size]
            output = self.accelerator.infer(batch)
            outputs.append(output)
        
        return torch.cat(outputs)

    def optimize(self, example_input: torch.Tensor, method: str = 'jit'):
        if method == 'tensorrt':
            self.accelerator.optimize_with_trt(example_input)
        else:
            self.accelerator.optimize_with_jit(example_input)

    def benchmark(self, input_data: torch.Tensor, n_runs: int = 100) -> dict:
        return self.accelerator.benchmark(input_data, n_runs)