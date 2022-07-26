import fire

import torch

from .util import to_export
from .export import export
from .logger import *

# the purpose of multiprocessing is to recover when torch-mlir crashes in C code,
# which is likely since it is unstable nightly builds only as of July 2022

def main(log_level='INFO'):
    print(to_export)  
    for i in range(len(to_export)):
        try:
            name = to_export[i].__name__
            torch.multiprocessing.spawn(export, args=(i, log_level), join=True)
        except torch.multiprocessing.ProcessExitedException:
            FAIL(f"torch-mlir: failed to export {name} (hard crashed)")
        except torch.multiprocessing.ProcessRaisedException:
            FAIL(f"torch-mlir: failed to export {name} (unhandled exception)")
            raise

if __name__=='__main__':
    fire.Fire(main)
    # main()