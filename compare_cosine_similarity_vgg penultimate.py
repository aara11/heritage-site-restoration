import numpy as np
import torch
from scipy import spatial
import torch.nn as nn
import os
from torchvision import models
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

model = models.vgg19(pretrained=True)


class VGG19_bot(nn.Module):
    def __init__(self, model):
        super(VGG19_bot, self).__init__()
        self.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        self.features = nn.ModuleList(list(model.features)).eval()

    def forward(self, x):
        for ii, m in enumerate(self.features):
            x = m(x)
        h = x.reshape(1, 1, 1, -1)
        h = self.classifier(h)
        h = h.reshape(-1).data.numpy()[:]
        return h


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class tt(nn.Module):
    def __init__(self):
        super(tt, self).__init__()

        self.transform = transforms.Normalize(mean=mean, std=std)

    def forward(self, x):
        return self.transform(x)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


tr = tt()
model_bot = VGG19_bot(model)
x = torch.rand(1, 3, 224, 224)
h = model_bot(x)
print(h.shape)


def read_img(path):
    image = Image.open(path)
    image = image.resize((224, 224), Image.ANTIALIAS)
    x = TF.to_tensor(image)
    x1 = torch.unsqueeze(x - mean[0], 0)
    x3 = torch.unsqueeze(x - mean[1], 0)
    x2 = torch.unsqueeze(x - mean[2], 0)

    x1 = x1 / np.sqrt(np.sum(x1.data.numpy() ** 2))
    x2 = x2 / np.sqrt(np.sum(x2.data.numpy() ** 2))
    x3 = x3 / np.sqrt(np.sum(x3.data.numpy() ** 2))
    x = torch.cat([x1 * std[0], x2 * std[1], x3 * std[2]], dim=1)
    return x


original_folder_path = './data/dataset/validation_data/data/'
aae_folder_path = './data/aae_results/validation_data/data/'
rnn_folder_path = './data/pixel_rnn/validation_data/data/'

fnames = os.listdir(original_folder_path)
scores = {}
for fname in fnames:
    name_ori = original_folder_path + fname
    name_aae = aae_folder_path + fname
    name_rnn = rnn_folder_path + fname
    img_ori = read_img(name_ori)
    img_aae = read_img(name_aae)
    img_rnn = read_img(name_rnn)
    rnn_feature = model_bot(img_rnn)
    aae_feature = model_bot(img_aae)
    ori_feature = model_bot(img_ori)
    aae = 1 - spatial.distance.cosine(aae_feature, ori_feature)
    rnn = 1 - spatial.distance.cosine(rnn_feature, ori_feature)
    scores[fname] = (aae, rnn)

print(scores)
