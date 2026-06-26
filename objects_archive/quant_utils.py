import torch
import torch.nn as nn


# ESTA CLASE ES EL CORAZÓN DE LA CUANTIZACIÓN
class QuantizedLinear(nn.Module):
    def __init__(self, original_linear, w_int, scale):
        super().__init__()
        self.bias = original_linear.bias
        # Registramos los pesos y la escala como buffers para que 
        # se guarden con el modelo y se muevan a la GPU automáticamente
        self.register_buffer('w_int', w_int)
        self.register_buffer('scale', scale)
        self.bias = original_linear.bias
    
    def forward(self, x):
        # 1. Antes de convertir
        tipo_int = self.w_int.dtype
        # 2. La conversión
        #w_float = self.w_int.to(x.dtype) * self.scale
        w_float = self.w_int.to(torch.float16) * self.scale.to(torch.float16)
        # 3. Después de convertir
        tipo_float = w_float.dtype
        # Este print solo se ejecutará la primera vez para no saturar la consola
        if not hasattr(self, 'printed_once'):
            print(f"[DEBUG] Capa {self}: Conversión realizada | {tipo_int} -> {tipo_float}")
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
        # Capturamos estadísticas del tensor de entrada
        batch_min = x.detach().min()
        batch_max = x.detach().max()
        
        # Actualizamos los límites globales de esta capa
        if self.min_val > batch_min:
            self.min_val.fill_(batch_min)
            
        if self.max_val < batch_max:
            self.max_val.fill_(batch_max)

        '''if self.call_count % 50 == 0:
            print(f"[DEBUG] Observador {self.linear}: capturadas {self.call_count} muestras. Rango actual: [{self.min_val:.4f}, {self.max_val:.4f}]")'''
        
        return self.linear(x)
    


def inspect_weights(model):
    print(f"\n{'='*60}")
    print(f"{'CAPA':<30} | {'DTYPE':<10} | {'RANGO (MIN/MAX)':<15}")
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
    print("INSPECCIÓN DE RANGOS DE CALIBRACIÓN")
    print("="*50)
    for name, module in model.named_modules():
        if isinstance(module, CalibrationObserver):
            min_v = module.min_val.item()
            max_v = module.max_val.item()
            # Si el valor sigue siendo infinito, significa que la capa nunca vio datos
            status = "OK" if (min_v != float('inf') and max_v != float('-inf')) else "¡ADVERTENCIA: SIN DATOS!"
            print(f"Capa: {name:40} | Rango: [{min_v:7.2f}, {max_v:7.2f}] | Estado: {status}")
    print("="*50 + "\n")

def verificar_cuantizacion(model):
    from torchao.dtypes import AffineQuantizedTensor
    import torch
    
    print("\n" + "="*80)
    print(f"{'CAPA':<30} | {'TIPO BUFFER':<20} | {'DTYPE REAL'}")
    print("="*80)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            peso = module.weight
            if isinstance(peso, AffineQuantizedTensor):
                # Accedemos al tensor que contiene los bits empaquetados
                # En torchao, el buffer de datos suele estar en ._data o .tensor_impl
                buffer = peso._data if hasattr(peso, '_data') else peso
                
                # Intentamos mostrar el dtype del buffer interno
                dtype_real = getattr(buffer, 'dtype', 'Desconocido')
                tipo_obj = type(buffer).__name__
                
                print(f"{name:<30} | {tipo_obj:<20} | {dtype_real}")
            else:
                print(f"{name:<30} | [!] NO CUANTIZADO")
    print("="*80 + "\n")

def activar_debug_inferencia(self):
    def hook_fn(module, input, output):
        # input[0] es la activación (entrada), module.weight es el peso
        weight = module.weight
        
        # Obtenemos el nombre del objeto para ver si es el cuantizado
        tipo_peso = type(weight).__name__
        
        # Intentamos acceder al almacenamiento real (buffer) si es cuantizado
        data_type = getattr(weight, 'dtype', 'Desconocido')
        
        print(f"\n[DEBUG INFERENCIA] Capa: {module}")
        print(f"  Tipo de objeto peso: {tipo_peso}")
        
        # Aquí está la prueba: Si es un AffineQuantizedTensor, 
        # su almacenamiento interno es int8/int4, no float32.
        if hasattr(weight, '_data'):
            print(f"  Dtype del buffer interno (bits): {weight._data.dtype}")
        elif hasattr(weight, 'tensor_impl'):
             # En versiones modernas de torchao
             print(f"  Estructura interna: {type(weight.tensor_impl).__name__}")
    
    # Registramos el hook en la primera capa lineal que encontremos
    for name, module in self.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(hook_fn)
            break