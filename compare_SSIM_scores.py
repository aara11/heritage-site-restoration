import cv2
import os
from skimage.measure import compare_ssim

original_folder_path = './data/dataset/validation_data/data/'
aae_folder_path = './data/aae_results/validation_data/data/'
rnn_folder_path = './data/pixel_rnn/validation_data/data/'

fnames = os.listdir(original_folder_path)
scores = {}
for fname in fnames:
    name_ori = original_folder_path + fname
    name_aae = aae_folder_path + fname
    name_rnn = rnn_folder_path + fname
    img_aae = cv2.imread(name_aae, cv2.IMREAD_GRAYSCALE)
    img_ori = cv2.imread(name_ori, cv2.IMREAD_GRAYSCALE)
    img_rnn = cv2.imread(name_rnn, cv2.IMREAD_GRAYSCALE)
    (score_aae, _) = compare_ssim(img_aae, img_ori, full=True)
    (score_rnn, _) = compare_ssim(img_rnn, img_ori, full=True)
    scores[name_ori[4:-4]] = (score_aae, score_rnn)

print(scores)
