# YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## A little illustration of the homework:
  Because of the time limit of kernel running time(7 hours once) and the time limit of the GPU use time in Kaggle, we cannot run the training process for long time, and because the model size is about 200+MB, we cannot upload it to the github by normal way. However with the BIG-FILE github upload process(git lfs), we recognize there has some bit broken in the uploading process, so that it cannot to be load into the torch for the finetune process.  
  Thus we only trained the model for 13 epochs, and the dataset is splitted into training set and testing set with the proportion rate of 7:3.  
  Because of the time limited... We didnt have enouth time for training the whole dataset for further testing by the course TAs.  
  If more time given and we can deal with the problem of big-size file upload to github repository, we can try to train the dataset with a total epochs of 100, and we will check the process of training by looking out tensorboard to find the best model (check f1-score, mAP, precision and recall rate)  
  Thank you for reading the illustration of the machine learning homework above, you can try to run the testing process by see the file yolov3/test.py, there is a illstration of the usage of its configurations.
#### Zhang Xuanrui, Hu Yiran.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/noba1anc3/yolov3
    $ cd yolov3/
    $ sudo pip3 install -r requirements.txt
    
## Test
Evaluates the model.

    $ python3 test.py --

## Model
https://github.com/Noba1anc3/model-and-log
please put the model to the folder yolov3/checkpoints/

## Tensorboard Log File
https://github.com/Noba1anc3/model-and-log
if you want to see the log information, please put the log to the folder yolov3/logs and run the command "Tensorboard" below.

## Inference
Uses pretrained weights to make predictions on images. Below table displays the inference times when using as inputs images scaled to 256x256. The ResNet backbone measurements are taken from the YOLOv3 paper. The Darknet-53 measurement marked shows the inference time of this implementation on my 1080ti card.

    $ python3 detect.py

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```
$ tensorboard --logdir='logs' --port=6006
```

## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
