import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then scale to [0,1] before computing
  mean and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  data_list = os.listdir(dir_name)

  # load images
  image_cat = []
  for test_train in data_list:
    test_train_dir = os.listdir(dir_name+test_train)
    for image_folder in test_train_dir:
      image_dir = os.listdir(dir_name+test_train+"/"+image_folder)
      for image in image_dir:
        img = Image.open(dir_name+test_train+"/"+image_folder+"/"+image)
        img = np.asarray(img)
        img = np.reshape(img,(img.shape[0]*img.shape[1]))
        if len(image_cat) == 0:
          image_cat = img
        else:
          image_cat = np.concatenate((image_cat, img))

  # normalize the images to [0,1]
  image_cat = (image_cat - min(image_cat))/(max(image_cat)-min(image_cat))
  image_cat = np.reshape(image_cat, (image_cat.shape[0], 1))
  std_scaler = StandardScaler()
  std_scaler.fit(image_cat)

  # calculate mean and std
  mean = std_scaler.mean_
  print(mean)
  std = std_scaler.var_ ** 0.5
  print(std)
  ############################################################################
  # Student code end
  ############################################################################
  return mean, std


