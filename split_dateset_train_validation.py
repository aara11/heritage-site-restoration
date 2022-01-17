import numpy as np
from shutil import copyfile
from utils import *

source_dir = './data/training_data/training_64/data/'
images = os.listdir(source_dir)

np.random.shuffle(images)
training = images[:256]
validation = images[256:]

create_folder("./data/dataset/")
create_folder("./data/dataset/training_data/")
create_folder("./data/dataset/training_data/data/")
create_folder("./data/dataset/validation_data/")
create_folder("./data/dataset/validation_data/data/")

for img in training:
    copyfile(source_dir + img, "./data/dataset/training_data/data/" + img)
for img in validation:
    copyfile(source_dir + img, "./data/dataset/validation_data/data/" + img)
