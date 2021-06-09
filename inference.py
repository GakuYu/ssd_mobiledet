#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 使用 pb 文件进行推理
@Date: 2021/06/08 11:37:22
@Author: Gaku Yu
@version: 1.0
'''
import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import ImageUtils, load_label_map

pb_path = 'ssdlite_mobiledet_cpu_320x320_coco_2020_05_19/frozen_model/frozen_inference_graph.pb'
image_path = 'test-data/kite.jpg'
label_map_path = 'test-data/mscoco_label_map.txt'
result_img_path = 'test-data/result.jpg'
input_w, input_h = (320, 320)

label_map = load_label_map(label_map_path, append_bg=True)

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
solve_cudnn_error()

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        resized = cv.resize(img, (input_w, input_h))
        input = np.expand_dims(resized, axis=0)
        (boxes, scores, classes, count) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: input})
        result = ImageUtils.draw(img, boxes[0].tolist(), classes[0].tolist(), scores[0].tolist(), label_map, threshold=0.6)
        cv.imwrite(result_img_path, cv.cvtColor(result, cv.COLOR_RGB2BGR))
        