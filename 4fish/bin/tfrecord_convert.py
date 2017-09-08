#!/usr/bin/python3

import os
import io
import sys
import math
import tensorflow as tf
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image

path = '/home/chrisjan/project/models/object_detection'
os.chdir(path)
sys.path.append('..')
from object_detection.utils import dataset_util

base_dir = '/home/chrisjan/project/training/4fish/'
image_dir = base_dir + 'images'
xml_dir = base_dir + 'xml'
data_dir = base_dir + 'data'
train_record_file = data_dir + '/train.record'
test_record_file = data_dir + '/test.record'
num_train = 0.9
#num_train = 1

def create_tf_example(row):
    full_path = image_dir + '/' + row['filename']
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = row['filename'].encode('utf8')
    image_format = b'jpg'
    xmins = [row['xmin'] / width]
    xmaxs = [row['xmax'] / width]
    ymins = [row['ymin'] / height]
    ymaxs = [row['ymax'] / height]
    classes_text = [row['class'].encode('utf8')]
    classes = [1]
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def xml_to_df():
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_list = []
    for xmlfile in os.listdir(xml_dir):
        xml_path = xml_dir + '/' + xmlfile
        #print(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_file = root.find('filename').text
        width = int(root.find('size')[0].text)
        height = int(root.find('size')[1].text)
        for member in root.findall('object'):
            class_name = member[0].text
            xmin = int(member[4][0].text)
            ymin = int(member[4][1].text)
            xmax = int(member[4][2].text)
            ymax = int(member[4][3].text)
            xml_list.append([image_file, width, height, class_name, xmin, ymin, xmax, ymax])
    train_df = pd.DataFrame(xml_list[0:math.floor(len(xml_list) * num_train)], columns=column_name)
    test_df = pd.DataFrame(xml_list[math.floor(len(xml_list) * num_train):], columns=column_name)
    return train_df, test_df

train_examples, test_examples = xml_to_df()

#build train_record
writer_train = tf.python_io.TFRecordWriter(train_record_file)
for index, row in train_examples.iterrows():
    train_example = create_tf_example(row)
    writer_train.write(train_example.SerializeToString())
writer_train.close()
#build test_record
writer_test = tf.python_io.TFRecordWriter(test_record_file)
for index, row in test_examples.iterrows():
    test_example = create_tf_example(row)
    writer_test.write(test_example.SerializeToString())
writer_test.close()
