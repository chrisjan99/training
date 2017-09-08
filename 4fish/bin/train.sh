#!/bin/bash
cd /chrisjan/project/models
python3 object_detection/train.py --logtostderr --pipeline_config_path=/chrisjan/project/training/4fish/data/faster_rcnn_inception_resnet_v2_atrous_pets.config --train_dir=/chrisjan/project/training/4fish/models/
