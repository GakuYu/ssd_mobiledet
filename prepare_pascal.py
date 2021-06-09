#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: 将 labelImg xml 标注数据文件按 pascal 的数据目录放置
@Date: 2021/06/07 15:40:07
@Author: Gaku Yu
@version: 1.0
'''
import shutil
import os
import sys
import logging
from xml.etree.ElementTree import ElementTree

logging.basicConfig(format="[ %(levelname)s | %(asctime)s ] %(message)s", level=logging.INFO, stream=sys.stdout)


data_path = 'test-data'
voc_img_path = 'VOC2007/JPEGImages/'
img_path = os.path.join(data_path, voc_img_path)
anno_path = os.path.join(data_path, 'VOC2007/Annotations/')
txt_path = os.path.join(data_path, 'VOC2007/ImageSets/Main/')

src_img_path = os.path.join(data_path, 'images')
src_anno_path = os.path.join(data_path, 'annotations')


[None if os.path.isdir(_p) else os.makedirs(_p) for _p in [img_path, anno_path, txt_path]]

fullpath_txt = open(os.path.join(data_path, 'fullpath.txt'), 'w')

train_txt = open(os.path.join(txt_path, 'train.txt'), 'w')
val_txt = open(os.path.join(txt_path, 'val.txt'), 'w')
trainval_txt = open(os.path.join(txt_path, 'trainval.txt'), 'w')
test_txt = open(os.path.join(txt_path, 'test.txt'), 'w')

anno_index = 0

for anno in os.listdir(src_anno_path):
    logging.info(anno)
    anno_file = os.path.join(src_anno_path, anno)
    anno_tree = ElementTree()
    anno_tree.parse(anno_file)
    img_full_name = anno_tree.findtext("filename")
    img_file = os.path.join(src_img_path, img_full_name)
    
    shutil.copy(img_file, os.path.join(img_path, img_full_name))
    anno_tree.find("folder").text = 'VOC2007'
    if anno_tree.find("path"):
        anno_tree.find("path").text = voc_img_path + img_full_name
    anno_tree.write(os.path.join(anno_path, anno), encoding="utf-8")
    
    anno_name = anno.split('.')[0]
    fullpath_txt.write("%s\n" % anno_name)
    trainval_txt.write("%s\n" % anno_name)
    # 拆分训练集与验证集
    if anno_index % 5 > 0:
        train_txt.write("%s\n" % anno_name)
    else:
        val_txt.write("%s\n" % anno_name)
    anno_index += 1

fullpath_txt.close()
train_txt.close()
val_txt.close()
trainval_txt.close()
test_txt.close()


