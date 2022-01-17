from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

from skimage import io
from utils import *

EPS = 1e-15


# Plots an image in a 3D axis space.
# first plots all pixel values mapped to log scale
# second plots log vaues of each pixel as:
# (log(pixel) - log(min_pixel))/ (log(max_pixel) - log(min_pixel))


def main():
    folder_paths = ['./data/aae_results/validation_data/', './data/dataset/validation_data/']
    create_folder("./data/3D_plot/")
    for folder_path in folder_paths:
        save_folder = folder_path.replace("./data", "./data/3D_plot").replace("validation_data/", "")
        create_folder(save_folder)
        file_lst = os.listdir(folder_path + "data/")
        for file_name in file_lst:
            img = io.imread(folder_path + 'data/' + file_name)
            x, y = np.mgrid[0:64, 0:64]
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(x, y, img, rstride=2, cstride=2, cmap=cm.coolwarm)
            ax.view_init(elev=40, azim=45)
            plt.savefig(save_folder + file_name)
            plt.clf()
            d_log = np.log(img + EPS)
            th = np.min(d_log)
            thm = np.max(d_log)
            d_log = (d_log - th) / (thm - th)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(x, y, d_log, rstride=2, cstride=2, cmap=cm.coolwarm)
            ax.view_init(elev=40, azim=45)
            plt.savefig(save_folder + file_name[:-4] + '_log.jpg')
            plt.clf()
            plt.close()


main()
