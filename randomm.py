# coding=UTF-8

import os
#from logzero import logger

train_txt = './data/custom/train.txt'
valid_txt = './data/custom/valid.txt'

with open(valid_txt, 'r') as f:
    lines = f.readlines()
    f.close()

with open(valid_txt, 'w') as f:
    for i in range(len(lines)):
        line = lines[i][19:-5]+'\n'
        f.write(line)
    f.close()
