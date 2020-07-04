import numpy as np
import cv2
import argparse
import os
import random

shape=[0,0,3]

min_width=64
min_height=64
max_width=128
max_height=128

parser=argparse.ArgumentParser()
parser.add_argument('--data_dir',default='',type=str)
args = parser.parse_args()

items=os.listdir(args.data_dir)
for item in items:
    if "_input." in item:
        target=cv2.imread(args.data_dir+"/"+item)
        shape=target.shape
        freq=random.randint(20,40)
        for i in range(0,freq):
            img=np.zeros(shape=shape,dtype=np.uint8)
            height=random.randint(min_height,max_height)
            width = random.randint(min_width, max_width)
            pos_x=  random.randint(0,shape[1]-width-1)
            pos_y= random.randint(0,shape[0]-height-1)
            img[pos_y:pos_y+height,pos_x:pos_x+width,:]=255
        cv2.imwrite(args.data_dir+"/"+item[:-10]+"_edge"+item[-4:],img)
        print(item)
