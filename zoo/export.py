from pathlib import Path
import pickle

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
    assert result.shape==mod.output_shape
    # DEBUG(result)

    ### torchscript export
    script_mod = torch.jit.script(mod)
    torch.jit.save(script_mod, ts_path)
    DONE(f'torchscript: jitted module {name} to {ts_path}')

    ### ONNX export
    try:
        torch.onnx.export(mod, inp, onnx_path, verbose=False)
        DONE(f'onnx: exported module {name} to {onnx_path}')
    except torch.onnx.errors.UnsupportedOperatorError:
        FAIL(f'onnx: exporting {name} to {onnx_path} failed (unsupported operator)')

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
        FAIL(f'torch_mlir: exporting {name} to {tosa_path} failed (compiler error')
        DEBUG(e.value)
        