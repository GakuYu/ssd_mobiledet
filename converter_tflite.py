#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 转换为 tflite 文件
@Date: 2021/06/08 10:21:05
@Author: Gaku Yu
@version: 1.0
'''
import os
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('pb_path', None, 'Frozen model path') 
tf.app.flags.DEFINE_string('save_dir', None, 'Tflite model save dir')
tf.app.flags.DEFINE_bool(
    'quantize', False,
    'To quantize the model')
tf.app.flags.mark_flag_as_required("pb_path")
tf.app.flags.mark_flag_as_required("save_dir")

def main(unuse_args):
    model_to_be_quantized = FLAGS.pb_path
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
                    graph_def_file=model_to_be_quantized, 
                    input_arrays=['normalized_input_image_tensor'], 
                    output_arrays=['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'], 
                    input_shapes={'normalized_input_image_tensor': [1, 320, 320, 3]}
                )
    converter.allow_custom_ops = True
    if FLAGS.quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_path = os.path.join(FLAGS.save_dir, "model_quant.tflite" if FLAGS.quantize else "model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

# python converter_tflite.py \
#   --pb_path=frozen_model_tflite/tflite_graph.pb \
#   --save_dir=frozen_model_tflite \
#   --quantize=True
if __name__ == '__main__':
    tf.app.run()