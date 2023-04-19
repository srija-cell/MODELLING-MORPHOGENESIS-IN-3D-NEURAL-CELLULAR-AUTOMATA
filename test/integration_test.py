import unittest
from main import return_zLayer_t,test_seed,load_data,create_mse_loss
import numpy as np
import tensorflow as tf
import csv
import os

class TestIntegration(unittest.TestCase):
    #checking inout data and returning the dimensions
    csv_file = "test_file_3x3x6.csv"
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
    arr, dimension = load_data(csv_file)
    assert dimension == [3, 3, 6], "Unexpected dimensions of loaded data."
    os.remove(csv_file)

    #test for stacking up z_layers
    SIZE_X = 3
    SIZE_Y = 3
    SIZE_Z = 6
    arr1 = np.array([[0, 0, 1, 255, 0, 0],
                    [1, 1, 1, 0, 255, 0],
                    [2, 2, 2, 0, 0, 255]])
    z_l1 = 2
    R1, G1, B1, Z1 = return_zLayer_t(arr1, z_l1)
    expected_R1 = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
    expected_G1 = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
    expected_B1 = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 255]])
    expected_Z1 = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]])
    assert np.array_equal(R1, expected_R1), f"Test case 1 failed: R1 = {R1}, expected_R1 = {expected_R1}"
    assert np.array_equal(G1, expected_G1), f"Test case 1 failed: G1 = {G1}, expected_G1 = {expected_G1}"
    assert np.array_equal(B1, expected_B1), f"Test case 1 failed: B1 = {B1}, expected_B1 = {expected_B1}"
    assert np.array_equal(Z1, expected_Z1), f"Test case 1 failed: Z1 = {Z1}, expected_Z1 = {expected_Z1}"



    #test for defining seed for 3d
    Channel = "20"
    Channel = int(Channel)
    result_seed = test_seed()
    excepted_seed = (1,3,3,6,20)
    # Check the shape of the seed array
    assert np.all(result_seed == excepted_seed), "Unexpected shape of seed array."


    #testing for loss function
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([2, 3, 4, 5])
    expected_mse_loss = np.mean(np.square(y_true - y_pred))
    print(expected_mse_loss)
    mse_loss = create_mse_loss()
    observed_mse_loss = mse_loss(y_true, y_pred)
    assert np.allclose(expected_mse_loss, observed_mse_loss)

print("\n\nIntegration testing successful.")

if __name__ == '__main__':
    unittest.main()