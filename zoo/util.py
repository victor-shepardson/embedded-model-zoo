import logging
from pathlib import Path

from termcolor import colored

import torch

to_export = []
logging.basicConfig(level=logging.INFO)

INFO = lambda s: logging.info(s)
DONE = lambda s: logging.info(colored(s, 'green'))
FAIL = lambda s: logging.info(colored(s, 'yellow'))

def export(mod):
    """decorate a torch.nn.Module to export a torchscript file"""
    to_export.append(mod)

def _export(mod_cls):
    logging.info('='*80)
    ts_root = Path('torchscript')
    onnx_root = Path('onnx')

    for root in [ts_root, onnx_root]:
        root.mkdir(exist_ok=True)

    name = mod_cls.__name__

    INFO(f'constructing module {name}')
    mod = mod_cls()

    INFO(f'running module {name}')
    inp = torch.zeros(mod.input_shape)
    result = mod(inp)
    assert result.shape==mod.output_shape
    logging.debug(result)

    ts_path = ts_root / (name+'.ts')
    script_mod = torch.jit.script(mod)
    torch.jit.save(script_mod, root / name)
    DONE(f'torchscript: jitted module {name} to {ts_path}')

    try:
        onnx_path = onnx_root / (name+'.onnx')
        torch.onnx.export(mod, inp, onnx_path, verbose=False)
        DONE(f'onnx: exported module {name} to {onnx_path}')
    except torch.onnx.symbolic_registry.UnsupportedOperatorError:
        FAIL(f'onnx: exporting module {name} to {onnx_path} failed')


