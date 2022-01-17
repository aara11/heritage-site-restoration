from skimage import io, exposure, color
import cv2
from utils import *

path = './data/training_data'
folder_lst = [os.path.join(path, o) for o in os.listdir(path)
              if os.path.isdir(os.path.join(path, o))]

folder_lst = [s for s in folder_lst if s.split('/')[-1][:4] == 'DSC_']
# get the folders corresponding to 2 images we will be extracting raining data from
for folder in folder_lst:
    folder_name = folder.split('/')[-1]
    file_lst = os.listdir(folder)
    create_folder("./data/training_data/training_64/")
    create_folder("./data/training_data/training_64/data/")

    for file_name in file_lst:
        image_path = os.path.join(folder, file_name)
        image = io.imread(image_path)

        gray = color.rgb2grey(image)

        out = exposure.equalize_adapthist(gray, kernel_size=None, clip_limit=0.01, nbins=256)
        h, w = out.shape
        cropped_part = int((w - h) / 2)
        out = out[:, cropped_part:cropped_part + h]
        out = cv2.resize(src=out,
                         dsize=(64, 64),
                         interpolation=cv2.INTER_CUBIC)
        io.imsave('./data/training_data/training_64/data/' + file_name, out)
