# YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## A little illustration of the homework:
  Because of the time limit of kernel running time(7 hours once) and the time limit of the GPU use time in Kaggle, we cannot run the training process for long time, and because the model size is about 200+MB, we cannot upload it to the github by normal way. However with the BIG-FILE github upload process(git lfs), we recognize there has some bit broken in the uploading process, so that it cannot to be load into the torch for the finetune process.    
  Thank you for reading the illustration of the machine learning homework above, you can try to run the testing process by see the file yolov3/test.py, there is a illstration of the usage of its configurations.
#### Zhang Xuanrui, Hu Yiran.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/noba1anc3/yolov3
    $ cd yolov3/
    $ sudo pip3 install -r requirements.txt
    
## Test
Evaluates the model.

    $ python3 test.py 
    
## Model
  https://bhpan.buaa.edu.cn/#/link/810DBF5B1BEC3D5E7D62F7179FBA09D9%20Valid%20Until:%202020-02-05%2023:59
  Please put the model to the folder yolov3/checkpoints/  
  If the bhpan failed, please issue to the repository or send mail to 751978769@qq.com.  
