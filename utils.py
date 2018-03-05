import numpy as np
import os
import skimage.color
import skimage.io
import skimage.transform

def onehot(x):
    x_onehot=np.zeros(7)
    x_onehot[x]=1
    return x_onehot

def load_image(path):
    img=[]
    label=[]
    for filename in os.listdir(path):
        filepath=path+'/'+filename
        label.append(onehot(int(filename[0])))
        crop_images=skimage.color .gray2rgb(skimage.io.imread(filepath))/255.0
        #gray image to rgb image
        images=skimage.transform.resize(crop_images,(224,224))
        img.append(np.array(images))
    return img,label

def next_batch(x,y,batch_size):
    total_num=len(x)
    x_copy=np.array(x)
    y_copy=np.array(y)
    idx=np.arange(0,total_num)
    np.random.shuffle(idx)
    x_copy=x_copy[idx]
    y_copy=y_copy[idx]
    i=0
    while True:
        if i+batch_size<=total_num:
            x_batch=x_copy[i:i+batch_size]
            y_batch=y_copy[i:i+batch_size]
            yield x_batch,y_batch
            i+=batch_size
        else:
            i=0
            idx=np.arange(0,total_num)
            np.random.shuffle(idx)
            x_copy=x_copy[idx]
            y_copy=y_copy[idx]
            continue


