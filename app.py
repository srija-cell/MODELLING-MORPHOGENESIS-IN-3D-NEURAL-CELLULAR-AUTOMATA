#@title web interface
import matplotlib
matplotlib.use('Agg')
import gradio as gr
import os
import io
import PIL.Image, PIL.ImageDraw
import requests
import numpy as np
import matplotlib.pylab as pl
import glob
import tensorflow as tf
from IPython.display import Image, HTML, clear_output
import tqdm
import os
import csv
import numpy as np
from IPython.display import Image, HTML, clear_output
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Conv3D
from keras.models import Sequential, load_model


def train_model(csv_file, training_type, channels, training_iterations, batch_size, color):
  print(csv_file, training_type, channels, training_iterations, batch_size, color)

  path=csv_file.name

  def load_data(path):
    """Loads the data from a CSV file and returns it as a numpy array along with its dimensions"""
    file_name = path.split('/')[-1].split('.')[0]
    dimension = [int(x) for x in file_name.split("_")[-1].split("x")]
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    arr = np.array(data).astype(float)
    return arr, dimension

  def return_zLayer(arr, z):
      """Returns a single layer of voxels"""
      z_slice = np.array([arr[arr[:, 2] == z]])
      Zxy = z_slice[0][...,:6].astype(int)
      R = np.zeros((SIZE_X, SIZE_Y))
      G = np.zeros((SIZE_X, SIZE_Y))
      B = np.zeros((SIZE_X, SIZE_Y))
      Z = np.zeros((SIZE_X, SIZE_Y))
      for i in Zxy:
          Z[i[0]][i[1]] = 1
          R[i[0]][i[1]] = i[3]
          G[i[0]][i[1]] = i[4]
          B[i[0]][i[1]] = i[5]
      return R, G, B, Z

  def stack_layers(arr):
      """Stacks up all the layers of voxels"""
      r_stack, g_stack, b_stack, z_stack = return_zLayer(arr, 0)
      for i in range(1, SIZE_Z):
          r, g, b, a = return_zLayer(arr, i)
          r_stack = np.dstack((r_stack, r))
          g_stack = np.dstack((g_stack, g))
          b_stack = np.dstack((b_stack, b))
          z_stack = np.dstack((z_stack, a))
      return r_stack, g_stack, b_stack, z_stack
 
 #@title  Parameters
  arr, dimention = load_data(path)
  max_dim=int(max(dimention))
  SIZE_X=dimention[0]
  SIZE_Y=dimention[1]
  SIZE_Z=dimention[2]
 
  flag = True if training_type[1] == 1 else False
  BATCH_SIZE = batch_size
  FACE_COLOR=color
  Channel=int(channels)
  print(dimention)
  print(arr)  
  arr = np.array(arr).astype(float)

    
  r_stack,g_stack,b_stack,z_stack=stack_layers(arr)

  #@title Define target  

  target=np.stack((r_stack/255,g_stack/255,b_stack/255,z_stack),axis=-1)

  def plot_target(r_stack, g_stack, b_stack, z_stack, max_dim, SIZE_X, SIZE_Y, SIZE_Z, flag=True, FACE_COLOR=(0, 0, 1, 0.2)):
      z_resized = np.pad(z_stack, (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant')
      r_resized = np.pad(r_stack, (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant')
      g_resized = np.pad(g_stack, (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant')
      b_resized = np.pad(b_stack, (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant')
      target_resized = np.stack((r_resized/255, g_resized/255, b_resized/255, z_resized), axis=-1)
      fig_t = plt.figure()
      ax = fig_t.add_subplot(projection='3d')
      if flag:
          ax.voxels(target_resized[...,3], facecolors=target_resized[..., :3])
      else:
          ax.voxels(target_resized[...,3], facecolors=FACE_COLOR)
      ax.set_title('Target voxel')
      return fig_t


  fig_t = plot_target(r_stack, g_stack, b_stack, z_stack, max_dim, SIZE_X, SIZE_Y, SIZE_Z, flag, FACE_COLOR)

  def create_seed(channels, size_x, size_y, size_z):
    """
    This function creates a seed array of zeros with dimensions [1, size_x, size_y, size_z, channels] and sets
    the center of the array to 1 for the first three channels.
    
    Args:
    channels (int): The number of channels for the seed array.
    size_x (int): The size of the array along the x-axis.
    size_y (int): The size of the array along the y-axis.
    size_z (int): The size of the array along the z-axis.
    
    Returns:
    seed (numpy array): A seed array of zeros with dimensions [1, size_x, size_y, size_z, channels] and the center
    of the array set to 1 for the first three channels.
    """
    seed = np.zeros([1, size_x, size_y, size_z, channels], np.float32)
    seed[:, size_x//2, size_y//2, size_z//2, :3] = 1
    print("Shape of seed:")
    print(seed.shape)
    return seed
  #defien seed
  seed= create_seed(channels=Channel, size_x=SIZE_X, size_y=SIZE_Y, size_z=SIZE_Z)

  def get_living_mask(x):
    """Returns a mask that identifies living cells in the CA grid"""
    alpha = x[..., 3:4]
    return  tf.cast(tf.nn.max_pool3d(alpha,3,1,'SAME') > 0.1,tf.float32)

  class CA(tf.Module):
    """Defines the 3D Cellular Automaton model"""
    def __init__(self):
      self.model=tf.keras.Sequential([
          Conv3D(filters=Channel*3, kernel_size=3, padding='same',input_shape=(SIZE_X,SIZE_Y,SIZE_Z,Channel), activation=tf.nn.relu),
          Conv3D(filters=Channel, kernel_size=3, padding='same',kernel_initializer=tf.zeros),
      ])
    
    @tf.function
    def __call__(self,x):
      """Runs a forward pass of the model"""
      #alive_mask= get_living_mask(x)
      update_mask = tf.floor(tf.random.uniform(x.shape) + 0.5)
      x= x+self.model(x)*update_mask
      #x *= alive_mask
      return x

  ca=CA()
  x=seed
  ittiration=100
  for i in range(ittiration):
    x=ca(x)

  #@title Defining loss & train step
  #defining optimizerimport tensorflow as tf
  
  # Create a PiecewiseConstantDecay learning rate schedule
  def create_lr_schedule():
      """Creates a PiecewiseConstantDecay learning rate schedule"""
      boundaries = [1000] # Set boundaries for learning rate changes
      values = [1e-3, 3e-4] # Set learning rate values for each boundary
      return tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

  # Define the optimizer
  def create_optimizer(lr):
      """Creates an Adam optimizer"""
      return tf.optimizers.Adam(lr)

  # Define the mean squared error loss function
  def create_mse_loss():
      """Creates a mean squared error loss function"""
      return tf.keras.losses.MeanSquaredError()

  # Define the training step function
  @tf.function
  def training_step(seed, target, ca, BATCH_SIZE, mse_loss, trainer, flag):
      """Performs a single training step"""
      with tf.GradientTape() as g:
          # Repeat the seed BATCH_SIZE times to create a batch of seeds
          x = tf.repeat(seed, BATCH_SIZE, 0)
          N = 100 # Number of iterations
          for i in range(N):
              x = ca(x)
              if flag:
                  # Compute loss for RGBA channels
                  loss = mse_loss(x[..., :4], tf.cast(target, tf.float32))
              else:
                  # Compute loss for only alpha channel
                  loss = mse_loss(x[..., :4][...,3], tf.cast(target[...,3], tf.float32))  

          # Compute gradients and update the model parameters
          params = ca.trainable_variables
          grads = g.gradient(loss, params) 
          grads = [g / (tf.norm(g) + 1e-8) for g in grads]
          trainer.apply_gradients(zip(grads, params))
          return loss, x

  # Define the training loop
  def train_model(seed, target, ca, training_iterations, batch_size, flag):
      """Trains the 3D Cellular Automaton model"""
      # Create the optimizer, loss function, and learning rate schedule
      lr_schedule = create_lr_schedule()
      optimizer = create_optimizer(lr_schedule)
      mse_loss = create_mse_loss()

      # Track the loss over time
      loss_log = []

      for i in range(int(training_iterations)):
          # Take a training step
          loss, x = training_step(seed, target, ca, int(batch_size), mse_loss, optimizer, flag)
          
          # Append the loss to the log
          loss_log.append(loss.numpy())
          
          # Print the loss every 20 iterations
          if i % 20 == 0:
              print(i, loss.numpy(), flush=True)
              
      # Create a plot of the loss over time
      fig_l, ax = plt.subplots()
      ax.plot(loss_log, '.', alpha=0.3)
      ax.set_yscale('log')
      ax.set_title('Loss over Training Iterations')
      ax.set_xlabel('Training Iterations')
      ax.set_ylabel('Loss')
      plt.close(fig_l)
      
      return fig_l

  
  fig_l=train_model(seed, target, ca, training_iterations, batch_size, flag)

  temp = x[0][...,:4]
  max_v=np.max(temp)
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(projection='3d')
  temp = np.where(temp < 0, 0, temp)
  temp=temp/max_v
  z_resized = np.pad(temp[...,3], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant')
  r_resized = (np.pad(temp[...,0], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))
  g_resized = (np.pad(temp[...,1], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))
  b_resized = (np.pad(temp[...,2], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))

  obj=np.stack((r_resized,g_resized,b_resized,z_resized),axis=-1)


  xx = np.where(z_resized < 0.1, 0, z_resized)
  if flag:
    ax.voxels(xx, facecolors=obj[...,:3])
  else:
    ax.voxels(xx, facecolors=FACE_COLOR)

  x=seed
  ittiration="100"#@param[100,200,300]
  for i in range(int(ittiration)):
    x=ca(x)
    clear_output(True) 
    
    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot(projection='3d')
    temp = x[0][...,:4]
    max_v=np.max(temp) 
    temp = np.where(temp < 0, 0, temp)
    temp=temp/max_v
    z_resized = np.pad(temp[...,3], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant')
    r_resized = (np.pad(temp[...,0], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))
    g_resized = (np.pad(temp[...,1], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))
    b_resized = (np.pad(temp[...,2], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))

    obj=np.stack((r_resized,g_resized,b_resized,z_resized),axis=-1)
    
    if flag:
      ax.voxels(xx, facecolors=obj[...,:3])
    else:
      ax.voxels(xx, facecolors=FACE_COLOR)
    
    from mpl_toolkits.mplot3d import Axes3D
    ax.view_init(azim=i)

    filename = "img/voxel_plot_{}.png".format(i)
    plt.savefig(filename) 
    
  x=seed
  ittiration="100"#@param[100,200,300]
  for i in range(int(ittiration)):
    x=ca(x)
    clear_output(True) 
    if  i==50:
      msk = np.ones([1,SIZE_X,SIZE_Y,SIZE_Z,Channel],np.float32)
      msk[ :,:, :, SIZE_Z//2:, :] = 0
      x=x*msk
    
    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot(projection='3d')
    temp = x[0][...,:4]
    max_v=np.max(temp)
    temp = np.where(temp < 0, 0, temp)
    temp=temp/max_v
    z_resized = np.pad(temp[...,3], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant')
    r_resized = (np.pad(temp[...,0], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))
    g_resized = (np.pad(temp[...,1], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))
    b_resized = (np.pad(temp[...,2], (((max_dim-SIZE_X)//2,(max_dim-SIZE_X)//2), ((max_dim-SIZE_Y)//2,(max_dim-SIZE_Y)//2), ((max_dim-SIZE_Z)//2,(max_dim-SIZE_Z)//2)), mode='constant'))

    obj=np.stack((r_resized,g_resized,b_resized,z_resized),axis=-1)
    if i==int(ittiration):
      for i in range(20):
        fig = plt.figure(figsize=(10,10))
        ax=fig.add_subplot(projection='3d')
        xx = np.where(z_resized < 0.1, 0, z_resized)
        ax.voxels(xx, facecolors=obj[...,:3])
        ax.voxels(xx, facecolors='silver')
        
        from mpl_toolkits.mplot3d import Axes3D
        ax.view_init(azim=i)

        filename = "img/voxel_plot_{}.png".format(i)
        plt.savefig(filename)


    xx = np.where(z_resized < 0.1, 0, z_resized)
    
    if flag:
      ax.voxels(xx, facecolors=obj[...,:3])
    else:
      ax.voxels(xx, facecolors=FACE_COLOR)
    
    from mpl_toolkits.mplot3d import Axes3D
    ax.view_init(azim=i)

    filename = "img/voxel_plot_{}.png".format(i)
    plt.savefig(filename)

    #@title Convertingn image sequence to video 
  from moviepy.editor import ImageSequenceClip

  file_list = ["img/voxel_plot_{}.png".format(i) for i in range(int(ittiration))]

  clip = ImageSequenceClip(file_list, fps=6)
  clip.write_videofile("growth.mp4")

  output_video = "growth.mp4"

  return fig_t,fig_l,output_video



input_csv = gr.inputs.File(label='CSV File')
input_training_type = gr.inputs.Radio(choices=[('Structure only', 0), ('structure with color', 1)], label='Training Type',default=('Structure only', 0))
input_channels = gr.inputs.Number(label='Channels',default=16)
input_training_iterations = gr.inputs.Number(label='Training Iterations',default=1000)
input_batch_size = gr.inputs.Number(label='Batch Size',default=1)
input_color = gr.inputs.Dropdown(choices= ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'gray', 'silver', 'purple', 'orange', 'brown']
, label='Color',default='silver')

# output_video = gr.outputs.Video(label="Output Video")


# Create the Gradio interface
gr.Interface(fn=train_model,
             inputs=[input_csv, input_training_type, input_channels, input_training_iterations, input_batch_size, input_color],
             outputs=["plot","plot","video"],
             title='Loss Plot and Images Generator'
             ).launch(debug=True)