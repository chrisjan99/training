#!/bin/bash
cd /chrisjan/project/models
python3 object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path /chrisjan/project/training/4fish/data/faster_rcnn_inception_resnet_v2_atrous_pets.config --trained_checkpoint_prefix /chrisjan/project/training/4fish/models/model.ckpt-10000 --output_directory /chrisjan/project/training/4fish/output
