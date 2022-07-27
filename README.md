# embedded-model-zoo

this repo is a python package. running it as a script will define and export various neural network graphs to various formats.

these neural networks are not trained models and are somewhat arbitary in architecture. thus, absolute performance numbers are not meaningful. but they should provide a sense of the relative presence/performance of various operators between formats and runtimes.

`nn.Modules` defined in pytorch are exported to:
* torchscript (`.ts`) using `torch.jit` 
    * -> ONNX (`.onnx`) via torchscript using `torch.onnx`
        * -> tensorflow SavedModel (`.tf`) via ONNX using `onnx-tf`
            * -> tflite (`.tflite`) via SavedModel using `tensorflow.lite`
* TOSA (`.tosa`) using `torch-mlir`

## setup with torch-mlir and nightly pytorch on arm64 mac:

(torch-mlir needs rosetta, but tensorflow and pytorch nightly seem unable to coexist on rosetta)

```
# NOTE the env names matter

CONDA_SUBDIR=osx-64 conda create -n zoo-tf -c conda-forge python pip fire tensorflow 
conda activate zoo-tf
conda config --env --set subdir osx-64
pip install tensorflow-probability onnx-tf

CONDA_SUBDIR=osx-64 conda create -n zoo -c conda-forge python pip fire
conda activate zoo
conda config --env --set subdir osx-64
pip install --pre torch-mlir -f https://github.com/llvm/torch-mlir/releases --extra-index-url https://download.pytorch.org/whl/nightly/cpu

cd embedded_model_zoo
pip install -e .
python -m zoo
```

may need this workaround:

https://github.com/llvm/torch-mlir/issues/853#issuecomment-1148237757

`python -m zoo --log_level=DEBUG` to see errors

## adding models

define new `nn.Module` subclasses in `zoo/models/*.py`, using the `@register` decorator. If adding new files under `models/`, be sure to import them from `models/__init__.py` 

## notes

- tflite and TOSA conversion often choke on `unsqueeze` ops (which torchscript seems to use when indexing with `None` e.g. `x[...,None]` to add a new final dimension to a tensor. workaround is to use `reshape` in pytorch code instead)

