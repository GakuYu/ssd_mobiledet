#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: ssd_mobiledet 训练
@Date: 2021/06/07 17:30:26
@Author: Gaku Yu
@version: 1.0
'''
import tensorflow.compat.v1 as tf
from object_detection.model_main import main

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if __name__ == '__main__':
    solve_cudnn_error()
    tf.app.run()