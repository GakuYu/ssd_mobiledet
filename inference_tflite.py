#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 使用 tflite 文件进行推理
@Date: 2021/06/08 11:37:35
@Author: Gaku Yu
@version: 1.0
'''
import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import ImageUtils, load_label_map

tflite_path = '/tf/ssd_mobiledet/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model_tflite/model.tflite'
is_quantized = False
# tflite_path = '/tf/ssd_mobiledet/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model_tflite/model_quant.tflite'
# is_quantized = True
image_path = 'test-data/kite.jpg'
label_map_path = 'test-data/mscoco_label_map.txt'
result_img_path = 'test-data/result_tflite.jpg'
input_w, input_h = (320, 320)

label_map = load_label_map(label_map_path, append_bg=False)

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
resized = cv.resize(img, (input_w, input_h))
input = np.expand_dims(resized, axis=0)
if is_quantized:
    input = np.array(input, dtype=np.float32)
else:
    input = np.array((input - 127.5)/127.5, dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input)
interpreter.invoke()
boxes = interpreter.get_tensor(output_details[0]['index'])
classes = interpreter.get_tensor(output_details[1]['index'])
scores = interpreter.get_tensor(output_details[2]['index'])
count = interpreter.get_tensor(output_details[3]['index'])
result = ImageUtils.draw(img, boxes[0].tolist(), classes[0].tolist(), scores[0].tolist(), label_map, threshold=0.6)
cv.imwrite(result_img_path, cv.cvtColor(result, cv.COLOR_RGB2BGR))

