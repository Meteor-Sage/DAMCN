import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.corrosion import corrosion_2classes as corrosion



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
# CORROSION_WEIGHTS_PATH = "/home/lym/IronErosionDrone/Mask_RCNN-master/logs/corrosion20220324T1729/mask_rcnn_corrosion_0258.h5"  # TODO: update this path
CORROSION_WEIGHTS_PATH = "../../logs/mask_rcnn_corrosion_0499.h5"  # TODO: update this path
# CORROSION_WEIGHTS_PATH = "/media/lym/0FC819220FC81922/金属腐蚀训练文件/corrosion20220324T1729/mask_rcnn_corrosion_0252.h5"  # TODO: update this path

config = corrosion.CorrosionConfig()
CORROSION_DIR = os.path.join(ROOT_DIR, "datasets/full_data")
# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# config.display()
# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# Load validation dataset
dataset = corrosion.CorrosionDataset()
# dataset.load_corrosion(CORROSION_DIR, "train")
dataset.load_corrosion(CORROSION_DIR, "val")
# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set path to balloon weights file

# Download file from the Releases page and set its path
# https://github.com/matterport/Mask_RCNN/releases
weights_path = CORROSION_WEIGHTS_PATH

# Or, load the last model you trained
# weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

resultList = {}
mean_iou_list = []
iou_list= []
id_list=[]
from mrcnn import metrics
from sklearn import metrics as met
import json

for id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, id, use_mini_mask=False)
    print("id:",id)
    results = model.detect([image], verbose=1)
    r = results[0]
    mask1 = np.zeros(gt_mask.shape[:1], dtype='int32')
    mask2 = np.zeros(gt_mask.shape[:1], dtype='int32')
    for i in range(r['masks'].shape[-1]):
        mask1 = mask1 | (r['masks'][:, :, i] * r['class_ids'][i])
    for i in range(gt_mask.shape[-1]):
        mask2 = mask2 | (gt_mask[:, :, i] * gt_class_id[i])
    resultTemp = met.classification_report(mask2.flatten(), mask1.flatten(), output_dict=True)

    mean_iou_list.append(metrics.mean_iou(mask1, mask2, 3))
    iou_list.append(metrics.iou(mask1, mask2, 2))
    info = dataset.image_info[id]
    id_list.append(info["id"])
    resultList[info["id"]]=resultTemp
tf = open("resultList.json", "w")
json.dump(resultList,tf)
tf.close()
np.savetxt("mean_iou_list.txt",mean_iou_list)
np.savetxt("iou_list.txt",iou_list)







# image_id = random.choice(dataset.image_ids)
image_id = 10
image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       dataset.image_reference(image_id)))

# Run object detection
results = model.detect([image], verbose=1)

# Display results
ax = get_ax(1)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

ax = get_ax(1)
visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                            dataset.class_names, [1.0 for i in range(gt_mask.shape[1])], ax=ax,
                            title="label")

from mrcnn import metrics
from sklearn import metrics as met
def mask_iou(mask1, mask2):
    area1 = mask1.sum()
    area2 = mask2.sum()
    inter = ((mask1+mask2)==2).sum()
    mask_iou = inter / (area1+area2-inter)
    return mask_iou

mask1 = np.zeros(gt_mask.shape[:1], dtype='int32')
mask2 = np.zeros(gt_mask.shape[:1], dtype='int32')
for i in range(r['masks'].shape[-1]):
    mask1 = mask1 | (r['masks'][:, :, i] * r['class_ids'][i])
for i in range(gt_mask.shape[-1]):
    mask2 = mask2 | (gt_mask[:, :, i] * gt_class_id[i])

result=met.classification_report(mask2.flatten(), mask1.flatten(),output_dict=True)
aaa=result['0']['precision']
print('acc:\t{}'.format(met.accuracy_score(mask2.flatten(), mask1.flatten())))
print('precison:\t{}'.format(met.precision_score(mask2.flatten(), mask1.flatten())))
print('mean_iou:\t{}'.format(metrics.mean_iou(mask1, mask2, 3)) )
print('iou:\t{}'.format(metrics.iou(mask1, mask2, 3)) )

##print('acc:\t{}'.format(metrics.compute_acc(mask1, mask2)) )
##print('recall:\t{}'.format(metrics.compute_recall(mask1, mask2)[0]) )
##print('f1:\t{}'.format(metrics.compute_f1_score(mask1, mask2)) )