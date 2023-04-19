import numpy as np
import csv
import tensorflow as tf

#checking inout data and returning the dimensions
def load_data(path):
    """Loads the data from a CSV file and returns it as a numpy array along with its dimensions"""
    file_name = path.split('/')[-1].split('.')[0]
    dimension = [int(x) for x in file_name.split("_")[-1].split("x")]
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    arr = np.array(data).astype(float)
    return arr, dimension

#stacking up z_layers
SIZE_X = 3
SIZE_Y = 3
SIZE_Z = 6
def return_zLayer_t(arr,z_l):
    z= np.array([arr[arr[:, 2] ==  z_l]])
    Zxy=z[0][...,:6].astype(int)

    Z=np.zeros((SIZE_X,SIZE_Y))
    R=np.zeros((SIZE_X,SIZE_Y))
    G=np.zeros((SIZE_X,SIZE_Y))
    B=np.zeros((SIZE_X,SIZE_Y))

    increment=0

    for i in Zxy:
      Z[i[0]][i[1]]=1
      R[i[0]][i[1]]=i[3]
      G[i[0]][i[1]]=i[4]
      B[i[0]][i[1]]=i[5]
    return R,G,B,Z

#test for defining seed
def test_seed():
    Channel = "20" #@param {type:"string"}
    Channel=int(Channel)
    seed = np.zeros([1,SIZE_X,SIZE_Y,SIZE_Z,Channel],np.float32)
    seed[:,SIZE_X//2,SIZE_Y//2,SIZE_Z//2,:3]=1
    return seed.shape


#testing for loss function
  # Define the mean squared error loss function
y_true = np.array([1, 2, 3, 4])
y_pred = np.array([2, 3, 4, 5])
def create_mse_loss():
    """Creates a mean squared error loss function"""
    return tf.keras.losses.MeanSquaredError()

