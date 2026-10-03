import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset
import os
from PIL import Image

# KEY CLASS FOR QUANTIZATION
class QuantizedLinear(nn.Module):
    def __init__(self, original_linear, w_int, scale):
        super().__init__()
        self.bias = original_linear.bias
        # We record the weights and scale as buffers so that 
        # be stored with the model and automatically moved to the GPU
        self.register_buffer('w_int', w_int)
        self.register_buffer('scale', scale)
        self.bias = original_linear.bias
    
    def forward(self, x):
        # 1. Before weight converting
        tipo_int = self.w_int.dtype
        # 2. conversion
        #w_float = self.w_int.to(x.dtype) * self.scale
        w_float = self.w_int.to(torch.float16) * self.scale.to(torch.float16)
        # 3. After converting
        tipo_float = w_float.dtype
        # This print statement will only execute the first time, so as not to clutter the console.
        if not hasattr(self, 'printed_once'):
            print(f"[DEBUG] Layer {self}: Conversion completed | {tipo_int} -> {tipo_float}")
            self.printed_once = True
        bias_fp16 = self.bias.to(torch.float16) if self.bias is not None else None
            
        #return torch.nn.functional.linear(x, w_float, self.bias)
        return torch.nn.functional.linear(x.to(torch.float16), w_float, bias_fp16)

class CalibrationObserver(nn.Module):
    def __init__(self, original_linear):
        super().__init__()
        self.linear = original_linear
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        # We capture statistics from the input tensor
        batch_min = x.detach().min()
        batch_max = x.detach().max()
        
        # We are updating the global limits for this layer.
        if self.min_val > batch_min:
            self.min_val.fill_(batch_min)
            
        if self.max_val < batch_max:
            self.max_val.fill_(batch_max)

        '''if self.call_count % 50 == 0:
            print(f"[DEBUG] Observer {self.linear}: {self.call_count} samples captured. Actual range: [{self.min_val:.4f}, {self.max_val:.4f}]")'''
        
        return self.linear(x)
    


def inspect_weights(model):
    print(f"\n{'='*60}")
    print(f"{'LAYER':<30} | {'DTYPE':<10} | {'RANGE (MIN/MAX)':<15}")
    print(f"{'-'*60}")
    
    for name, module in model.named_modules():
        if hasattr(module, 'w_int'):
            w_int = module.w_int
            dtype = w_int.dtype
            is_integer = (w_int.to(torch.float32) % 1 == 0).all().item()
            min_val = w_int.min().item()
            max_val = w_int.max().item()
            
            print(f"{name:<30} | {str(dtype):<10} | [{min_val}, {max_val}] | Int?: {is_integer}")
    print(f"{'='*60}\n")


def inspect_calibration_stats(model):
    print("\n" + "="*50)
    print("CALIBRATION RANGE INSPECTION")
    print("="*50)
    for name, module in model.named_modules():
        if isinstance(module, CalibrationObserver):
            min_v = module.min_val.item()
            max_v = module.max_val.item()
            # If the value remains infinite, it means the layer never saw any data.
            status = "OK" if (min_v != float('inf') and max_v != float('-inf')) else "¡WARNING: NO DATA!"
            print(f"Layer: {name:40} | Range: [{min_v:7.2f}, {max_v:7.2f}] | Status: {status}")
    print("="*50 + "\n")

def verificar_cuantizacion(model):
    from torchao.dtypes import AffineQuantizedTensor
    import torch
    
    print("\n" + "="*80)
    print(f"{'LAYER':<30} | {'BUFFER TYPE':<20} | {'REAL DTYPE'}")
    print("="*80)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            peso = module.weight
            if isinstance(peso, AffineQuantizedTensor):
                # We access the tensor containing the packed bits.
                # In Torchao, the data buffer is usually located in ._data or .tensor_impl.
                buffer = peso._data if hasattr(peso, '_data') else peso
                
                # We attempt to show the dtype of the internal buffer.
                dtype_real = getattr(buffer, 'dtype', 'Unknown')
                tipo_obj = type(buffer).__name__
                
                print(f"{name:<30} | {tipo_obj:<20} | {dtype_real}")
            else:
                print(f"{name:<30} | [!] UNQUANTIZED")
    print("="*80 + "\n")

def activar_debug_inferencia(self):
    def hook_fn(module, input, output):
        # input[0] is the activation (input), module.weight is the weight.
        weight = module.weight
        
        # We obtain the object's name to see if it is the quantized one.
        tipo_peso = type(weight).__name__
        
        # We attempt to access the actual storage (buffer) if it is quantized.
        data_type = getattr(weight, 'dtype', 'Desconocido')
        
        print(f"\n[DEBUG INFERENCE] Layer: {module}")
        print(f"  Object type weight: {tipo_peso}")
        
        # Here's the proof: If it's an AffineQuantizedTensor, 
        # Its internal storage is int8/int4, not float32.
        if hasattr(weight, '_data'):
            print(f" Internal buffer dtype (bits): {weight._data.dtype}")
        elif hasattr(weight, 'tensor_impl'):
             # En versiones modernas de torchao
             print(f"  Internal structure: {type(weight.tensor_impl).__name__}")
    
    # We register the hook on the first linear layer we find.
    for name, module in self.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(hook_fn)
            break