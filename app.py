import gradio as gr
import re
from tensorflow import keras
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.models import Sequential, load_model
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import ImageSequenceClip
import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import glob
import csv
import tensorflow as tf
from IPython.display import display
from IPython.display import Image, HTML, clear_output
import tqdm
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
clear_output()


# def save_csv_file(csv_file):
#     with open(csv_file.name, 'wb') as f:
#         f.write(csv_file.read())
#     return f"{csv_file.name} saved successfully!"

  # append new extension
def generate_output(csv_data_file):
    
    def to_rgba(x):
      return x[..., :4]

    def to_alpha(x):
      return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)

    def to_rgb(x):
      # assume rgb premultiplied by alpha
      rgb, a = x[..., :3], to_alpha(x)
      return 1.0-a+rgb

    def np2pil(a):
      if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1)*255)
      return PIL.Image.fromarray(a)

    def imwrite(f, a, fmt=None):
      a = np.asarray(a)
      if isinstance(f, str):
        fmt = f.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
          fmt = 'jpeg'
        f = open(f, 'wb')
      np2pil(a).save(f, fmt, quality=95)

    def imencode(a, fmt='jpeg'):
      a = np.asarray(a)
      if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = 'png'
      f = io.BytesIO()
      imwrite(f, a, fmt)
      return f.getvalue()


    def imshow(a, fmt='jpeg'):
        display(Image(data=imencode(a, fmt)))

    def tile2d(a, w=None):
        a = np.asarray(a)
        if w is None:
            w = int(np.ceil(np.sqrt(len(a))))
        th, tw = a.shape[1:3]
        pad = (w-len(a))%w
        a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
        h = len(a)//w
        a = a.reshape([h, w]+list(a.shape[1:]))
        a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
        return a

    def zoom(img, scale=4):
        img = np.repeat(img, scale, 0)
        img = np.repeat(img, scale, 1)
        return img

    class VideoWriter:
      def __init__(self, filename, fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

      def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
          h, w = img.shape[:2]
          self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
          img = np.uint8(img.clip(0, 1)*255)
        if len(img.shape) == 2:
          img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

      def close(self):
        if self.writer:
          self.writer.close()

      def __enter__(self):
        return self

      def __exit__(self, *kw):
        self.close()

    
    filename=csv_data_file.name
    print(filename)
    match = re.search(r"(\d+)x(\d+)x(\d+)", filename)

    dimention = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    print(dimention)
        #print(dimensions)  # Output: [29, 10, 31]
    # path = "/home/srija/Cellular_automata/gnca_/flagged/output/torus_19x6x19ngzgxu8o.csv"
    # path = csv_data_file.name
    with open(filename, 'r') as f:
      reader = csv.reader(f)
      data = list(reader)

    print(np.shape(data))

    SIZE_X=dimention[0]
    SIZE_Y=dimention[1]
    SIZE_Z=dimention[2]
    numpy_array = np.array(data)
    numpy_array = np.array(data[1:])
    arr=numpy_array[...,:3].astype(float)

    #print(arr)

    #@title Return a single layer of Z in bitmap representation
    def return_zLayer(arr,z):
        z= np.array([arr[arr[:, -1] ==  z]])
        Zxy=z[0][...,:2].astype(int)
        Z=np.zeros((SIZE_X,SIZE_Y))
        
        for i in Zxy:
            Z[i[0]][i[1]]=1   
        return Z

    #@title Creating array by stacking up x,y for all z
    z_stack=return_zLayer(arr,1)
    for i in range(1,SIZE_Z):
      x=return_zLayer(arr,i)
      z_stack = np.dstack((z_stack,x))
    #print(z_stack.shape)

    z_space = []
    for i in range(SIZE_Z):
        z_layer = return_zLayer(arr, i)
        z_space.append(list(z_layer))
    z_space=np.array(z_space)
    #print(z_space.shape)

    for i in range(SIZE_Z):
      imshow(z_space[i])

    dim=int(max(z_stack.shape))
    #print((dim-SIZE_Y))

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    padded_array = np.pad(z_stack, (((dim-SIZE_X)//2,(dim-SIZE_Z)//2), ((dim-SIZE_Y)//2,(dim-SIZE_Y)//2), ((dim-SIZE_Z)//2,(dim-SIZE_Z)//2)), mode='constant')
    ax.voxels(padded_array, edgecolor='k')

    fig.show()

    #@title defing seed for 2d
    Channel=SIZE_Z
    seed = np.zeros([1,SIZE_X,SIZE_Y,Channel],np.float32)
    seed[:,SIZE_X//2,SIZE_Y//2,Channel//2]=1
    # print(seed.shape)
    # print(seed[0].shape)

    #@title defining seed for 3d
    Channel=SIZE_Z
    seed = np.zeros([1,SIZE_X,SIZE_Y,SIZE_Z,1],np.float32)
    seed[:,SIZE_X//2,SIZE_Y//2,SIZE_Z//2]=1

    #@title Display seed
    temp = np.reshape(seed, (SIZE_X, SIZE_Y, SIZE_Z))
    # fig = plt.figure()
    # ax = fig.gca(111,projection='3d')

    padded_array = np.pad(temp, (((dim-SIZE_X)//2,(dim-SIZE_X)//2), ((dim-SIZE_Y)//2,(dim-SIZE_Y)//2), ((dim-SIZE_Z)//2,(dim-SIZE_Z)//2)), mode='constant')
    ax.voxels(padded_array, edgecolor='k')

    # fig.show()


    #@title Building Model  1


    class CA(tf.Module):
      def __init__(self):
        self.model=tf.keras.Sequential([
            Conv2D(128,3,padding='same',activation=tf.nn.relu),
            Conv2D(Channel,1,kernel_initializer=tf.zeros) 
        ])

      
      @tf.function
      def __call__(self,x):
        return x+self.model(x)


    #@title Building Model  (3d)
    choice='3d'#@param[2d,3d]


    class CA(tf.Module):
      def __init__(self):
        self.model=tf.keras.Sequential([
            Conv3D(filters=SIZE_Z*3,strides=1, kernel_size=3, padding='same', activation=tf.nn.relu),
            Conv3D(filters=SIZE_Z, kernel_size=3,padding='same',activation=tf.nn.relu),
            Conv3D(filters=1, kernel_size=1,padding='same', kernel_initializer=tf.zeros) 
        ])
      
      @tf.function
      def __call__(self,x):
        update_mask=tf.floor(tf.random.uniform(x.shape)+0.5)
        return x+self.model(x)*update_mask


    ca=CA()
    x=seed
    ittiration=10 #100
    for i in range(ittiration):
      x=ca(x)
      if i%20==0:
        clear_output(True)
        temp = tf.reshape(x,(SIZE_X, SIZE_Y, SIZE_Z))
        # imshow(zoom(temp[...,0]))
        # print('chanel 0')
        # imshow(zoom(temp[...,5]))
        # print('Channel 5')
      
        # imshow(zoom(temp[...,9]))
        # print('chanel 9')

    #@title Defining target 
    target_tensor = tf.expand_dims(tf.expand_dims(z_stack, axis=0), axis=-1)

    trainer=tf.optimizers.Adam(1e-3)

    for k in range(10):
      with tf.GradientTape() as g:
        x=seed
        ittiration =50
        for i in range(ittiration):
          x=ca(x)
          loss = tf.reduce_mean(tf.square(x - tf.cast(target_tensor, tf.float32)))

      params=ca.trainable_variables
      grads=g.gradient(loss, params) 
      #grads=[g/(tf.norm(g)+1e-8) for g in grads]
      trainer.apply_gradients(zip(grads,params))
      print(loss.numpy())

    #@title Loss function & train step

    #defining optimizer
    lr=tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000],[1e-3,3e-4])
    trainer=tf.optimizers.Adam(lr)
    loss_log=[]

    @tf.function
    def training_step():
      with tf.GradientTape() as g:
        x=seed
        n="10" #@param[10,50,100] #100
        ittiration =int(n)
        for i in range(ittiration):
          x=ca(x)
          loss = tf.reduce_mean(tf.square(x - tf.cast(target_tensor, tf.float32)))

      params=ca.trainable_variables
      grads=g.gradient(loss, params) 
      #grads=[g/(tf.norm(g)+1e-8) for g in grads]
      trainer.apply_gradients(zip(grads,params))
      return loss,x



    # imshow(zoom(z_space[0]))
    # imshow(zoom(z_space[5]))
    # imshow(zoom(z_space[9]))

    #@title Training loop {vertical-output:true}
    ittiration="10"#@param[1000,2000,5000] #100  #changed from 1000 to 100 for fast training
    for i in range(10): #100
      loss,x=training_step()
        
      loss_log.append(loss.numpy())
      if i%20==0:
        clear_output(True)
        temp = tf.reshape(x,(SIZE_X, SIZE_Y, SIZE_Z))
        imshow(zoom(temp[...,0]))
        print('chanel 0')
        imshow(zoom(temp[...,5]))
        print('Channel 5')
        imshow(zoom(temp[...,9]))
        print('chanel 9')
        print(type(loss_log))
        # plt.plot(loss_log,'.',alpha=0.3)
        # plt.yscale('log')
        # plt.show()
        print(i,loss.numpy(),flush=True)
    #--------------------------------------------------------------------------------------------------------


        
        
    #---------------------------------------------------------------------------------------------------------
    #test


    x=seed
    ittiration=20 #200
    for i in range(ittiration):
      x=ca(x)
      clear_output(True)
      
      fig = plt.figure()
      ax = fig.add_subplot(projection='3d')
      temp = np.reshape(x, (SIZE_X, SIZE_Y, SIZE_Z))
      xx = np.where(temp < 0.1, 0, temp)

      padded_array = np.pad(xx, (((dim-SIZE_X)//2,(dim-SIZE_Z)//2), ((dim-SIZE_Y)//2,(dim-SIZE_Y)//2), ((dim-SIZE_Z)//2,(dim-SIZE_Z)//2)), mode='constant')
      ax.voxels(padded_array, edgecolor='k')
      

      ax.view_init(azim=i)

      filename = "imgs/voxel_plot_{}.png".format(i)
      plt.savefig(filename)


    file_list = ["imgs/voxel_plot_{}.png".format(i) for i in range(ittiration)]

    clip = ImageSequenceClip(file_list, fps=12)
    clip.write_videofile("growth2.mp4") 
    
    return "growth2.mp4"

input_csv = gr.inputs.File(label="Select CSV File")

output_video = gr.outputs.Video(label="Output Video")

iface = gr.Interface(fn=generate_output, inputs=input_csv, outputs=output_video, capture_session=True, title="Generate Video From CSV")

iface.launch()