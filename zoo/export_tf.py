"""this is run as a script for torch/tf incompatibility reasons"""

from pathlib import Path

import fire

import onnx
import tensorflow as tf
import onnx_tf

# couldn't get logs to appear from here, why?
# from zoo.logger import *

def main(onnx_path):
    name = Path(onnx_path).stem

    tf_root = Path('tf')
    tflite_root = Path('tflite')
    tf_root.mkdir(exist_ok=True)
    tflite_root.mkdir(exist_ok=True)

    tf_path = tf_root / (name+'.tf')
    tflite_path = tflite_root / (name+'.tflite')

    # INFO(f'tflite: loading {onnx_path}')
    onnx_mod = onnx.load(onnx_path)

    tf_mod = onnx_tf.backend.prepare(onnx_mod)
    tf_mod.export_graph(tf_path)
    # DONE(f'tf: exported SavedModel {name} to {tf_path}')

    tflite_mod = tf.lite.TFLiteConverter.from_saved_model(str(tf_path)).convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_mod)
    # DONE(f'tflite: exported flite {name} to {tflite_path}')

if __name__=='__main__':
    fire.Fire(main)