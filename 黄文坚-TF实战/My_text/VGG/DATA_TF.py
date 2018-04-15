# _*_ coding:utf-8 _*_


import tensorflow as tf
import sys

# path
data_dir = '/shixiaofeng/myown/My_Project_test/voc_data/VOCdevkit/VOC2007'
image_idx = '008541'
image_path = '%s/JPEGImages/%s.jpg' % (data_dir, image_idx)
annotation_path = '%s/Annotations/%s.xml' % (data_dir, image_idx)

# read file
image_data = tf.gfile.FastGFile(name=image_path, mode='rb').read()

# read annotation
import xml.etree.ElementTree as ET

# from defusedxml.ElementTree import parser
tree = ET.parse(annotation_path)
root = tree.getroot()
print(root)
size = root.find('size')
shape = [int(size.find('height').text),
         int(size.find('width').text),
         int(size.find('depth').text)]
print(shape)

# # Find annotations.
bboxes = []
labels = []
labels_text = []
difficult = []
truncated = []
for obj in root.findall('object'):
    label = obj.find('name').text
    print(label)
    labels.append(1)  # int(VOC_LABELS[label][0]) label对应的类别编号， 此处直接使用1， 没什么特殊含义。
    labels_text.append(label.encode('ascii'))

    if obj.find('difficult') is not None:
        difficult.append(int(obj.find('difficult').text))
    else:
        difficult.append(0)

    if obj.find('truncated') is not None:
        truncated.append(int(obj.find('truncated').text))
    else:
        truncated.append(0)

    bbox = obj.find('bndbox')
    bboxes.append((float(bbox.find('ymin').text) / shape[0],
                   float(bbox.find('xmin').text) / shape[1],
                   float(bbox.find('ymax').text) / shape[0],
                   float(bbox.find('xmax').text) / shape[1]
                   ))
print(bboxes)
