import cv2
import lmdb
import numpy as np
import tensorflow as tf
import os
from os.path import exists
import pickle
import skimage.transform
import math
from PIL import Image
import pdb
import time
tfgan = tf.contrib.gan
"""---------------------------------------------------------"""
"""Data preprocess"""


def centercrop(image, crop):
    row,col,_ = image.shape
    startx = row//2-(crop//2)
    starty = col//2-(crop//2)    
    return image[startx:startx+crop,starty:starty+crop,:]



def export_images(db_path, out_dir, total_num, sub_num):
    print('Exporting', db_path, 'to', out_dir)
    env = lmdb.open(db_path, map_size=1099511627776, max_readers=100, readonly=True)
    count = 1
    dir_name= out_dir+"-"+str(sub_num)+"-"+str(total_num)
    if not exists(dir_name):
        os.makedirs(dir_name)
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        store=[]
        for key, val in cursor:
            #pdb.set_trace()    
            #image = np.array(Image.open(val))
            image = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
            #pdb.set_trace()    
            rows = image.shape[0]
            cols = image.shape[1]
            downscale = min(rows / 64., cols / 64.)
            image = skimage.transform.pyramid_reduce(image, downscale)
            image=centercrop(image,64)
            image *= 255.
            image = image.astype("uint8")
            store.append(image)
            if (count!=0 and count%sub_num==0) or count==total_num:
                pathname='{}/data_store_{}-{}.pkl'.format(dir_name, math.ceil(count/sub_num), int(total_num/sub_num)+1)
                with open(pathname, 'wb') as f:
                    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
                store=[]

            if count % 5000 == 0:
                print('Finished', count, 'images')
            count += 1




"""visualization"""

def visualize_sample(batch_x, directory):
    directory=directory
    print('visualize_sample...')
    outfile = directory+'/sample.png'
    for i in range(8):
        concat=batch_x[i*10]
        for j in range(9):
            concat=np.concatenate((concat, batch_x[i*10+j+1]), axis=1)
        
        if i==0:
            img=concat
        else:
            img=np.concatenate((img, concat), axis=0)

    img*=128
    img+=128
    img=img.astype("uint8")
    cv2.imwrite(outfile, img)
    

def visualize_generator(train_step_num, start_time, data_np, dir):
    directory=dir
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    outfile = directory+'/train_%d.png' % (train_step_num)
    img=np.squeeze(data_np)
    img*=128
    img+=128
    img=img.astype("uint8")
    cv2.imwrite(outfile, img)






