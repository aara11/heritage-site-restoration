import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy.misc import imsave
import pickle
from collections import OrderedDict
import os
from aae import *


def inf_train_gen(train_gen):
    while True:
        for i, (images, _) in enumerate(train_gen):
            image = images[:, 0, :, :]
            image = image.unsqueeze(1)
            yield image


def get_transform():
    dim_change = lambda x: 2.0 * (x - 0.5)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(dim_change),
            #                    transforms.Normalize(mean=[0.5],std=[0.5]),
        ])
    return preprocess


def dataset_iterator(folder_name, batch_size=64):
    transform = get_transform()
    data = datasets.ImageFolder(folder_name, transform=transform)
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True,
                                              num_workers=0)
    return data_loader


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def load_pickle(filename):
    file = open(filename, 'rb')
    object_file = pickle.load(file)
    return object_file


def save_pickle(filename, fileobject):
    filehandler = open(filename, "wb")
    pickle.dump(fileobject, filehandler)
    filehandler.close()


def save_images(X, save_path, sub):
    for i in range(len(X)):
        img = X[i][0]
        img = img.detach().numpy()
        img = img + 1
        img = img * 0.5
        imsave(save_path + str(i) + sub + '.jpg', img)
    return


def read_aae_weights(epath=None, gpath=None, dpath=None):
    netE = netG = netD = None
    if epath is not None:
        emod = torch.load(epath)
        netE = Q_net()
        new_state_dict = OrderedDict()
        state_dict = emod['state_dict']
        for k, v in state_dict.items():
            name = k  # remove `module.`
            if 'num_batches_tracked' in k:
                continue
            new_state_dict[name] = v
        netE.load_state_dict(new_state_dict)  # gmod['state_dict'])

    if gpath is not None:
        gmod = torch.load(gpath)
        netG = P_net()
        new_state_dict = OrderedDict()
        state_dict = gmod['state_dict']
        for k, v in state_dict.items():
            name = k  # remove `module.`
            if 'num_batches_tracked' in k:
                continue
            new_state_dict[name] = v
        netG.load_state_dict(new_state_dict)

    if dpath is not None:
        dmod = torch.load(dpath)
        netD = D_net_gauss()
        netD.load_state_dict(dmod['state_dict'])
    return netE, netG, netD
