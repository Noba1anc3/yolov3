from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import time
import datetime
import argparse

from PIL import Image
from logzero import logger

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def test(img_path, anno_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3.pth")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names")
    parser.add_argument("--conf_thres", type=float, default=0.01)
    parser.add_argument("--nms_thres", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpu", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=416)
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights_path, map_location=device))
    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(img_path, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print('')
    logger.info("Performing object detection:")

    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds = current_time - prev_time)
        prev_time = current_time
        logger.info("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print('')
    logger.info("Saving:")

    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        logger.info("(%d) Image: '%s'" % (img_i+1, path))

        # Create plot
        img = np.array(Image.open(path))

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                x1 = round(x1.item(),2)
                x2 = round(x2.item(), 2)
                y1 = round(y1.item(), 2)
                y2 = round(y2.item(), 2)
                pos_str = str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n'
                pre_len = len(img_path)
                out_str = path[pre_len:-4] + ' ' + str(round(conf.item(),4)) + ' ' + pos_str
                cls = classes[int(cls_pred)]
                if cls == 'core':
                    with open('core.txt', 'a') as f:  # 设置文件对象
                        f.write(out_str)
                else:
                    with open('coreless.txt', 'a') as f:
                        f.write(out_str)

if __name__ == "__main__":
    test("data/test/image/",'')
