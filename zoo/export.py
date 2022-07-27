from pathlib import Path
import warnings
import subprocess
import sys

import torch

from . import models
from .util import to_export
from .logger import *

def export(_, i, log_level):
    logging.basicConfig(level=log_level)

    mod_cls = to_export[i]

    logging.info('='*80)
    ts_root = Path('torchscript')
    onnx_root = Path('onnx')
    tosa_root = Path('tosa')

    for root in [ts_root, onnx_root, tosa_root]:
        root.mkdir(exist_ok=True)

    name = mod_cls.__name__

    ts_path = ts_root / (name+'.ts')
    onnx_path = onnx_root / (name+'.onnx')
    tosa_path = tosa_root / (name+'.tosa')
    

    INFO(f'constructing module {name}')
    mod = mod_cls()
    mod.eval()

    INFO(f'running module {name}')
    inp = torch.zeros(mod.input_shape)
    result = mod(inp)
    shapes = (result.shape, mod.output_shape)
    assert all(s==shapes[0] for s in shapes), shapes
    # DEBUG(result)

    ### torchscript export
    script_mod = torch.jit.script(mod)
    torch.jit.save(script_mod, ts_path)
    DONE(f'torchscript: jitted module {name} to {ts_path}')

    ### ONNX export
    try:
        with warnings.catch_warnings():
            # ignore warnings about batch size
            warnings.simplefilter("ignore")
            torch.onnx.export(mod, inp, onnx_path, verbose=False)
        DONE(f'onnx: exported module {name} to {onnx_path}')

        ### tflite export
        python = sys.executable.replace('zoo', 'zoo-tf')
        result = subprocess.run([python, 'zoo/export_tf.py', onnx_path],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        DEBUG(result.stdout)
        DEBUG(result.stderr)

        DONE(f'tflite: exported {name}')

    except torch.onnx.errors.UnsupportedOperatorError as e:
        FAIL(f'onnx: exporting {name} to {onnx_path} failed (unsupported operator)')
        DEBUG(e)
        FAIL(f'tflite: exporting {name} not attempted (onnx failed)')


    except subprocess.CalledProcessError as e:
        FAIL(f'tflite: exporting {name} failed')
        DEBUG(e)
        DEBUG(e.stdout)
        DEBUG(e.stderr)
        

    return

    ### TOSA export
    try:
        import torch_mlir
        tosa_mod = torch_mlir.compile(mod, inp, output_type="tosa")
        # INFO(tosa_mod)
        # INFO(tosa_mod.operation.get_asm(large_elements_limit=10))
        with open(tosa_path, 'w') as f:
            f.write(str(tosa_mod))
            # f.write(tosa_mod.operation.get_asm(large_elements_limit=10))
        DONE(f'torch_mlir: exported module {name} to {tosa_path}')

    except ImportError:
        FAIL(f'torch_mlir: exporting {name} to {tosa_path} failed (torch-mlir not available)')    
    except torch_mlir.compiler_utils.TorchMlirCompilerError as e:
        FAIL(f'torch_mlir: exporting {name} to {tosa_path} failed (compiler error)')
        DEBUG(e.value)
        