from aae import *
from collections import OrderedDict
from torch.autograd import Variable
from utils import *

batch_size = 64
z_dim = 100


# generate fake images from the trained latent distribution

def main():
    gpath = './results/aae_models/P_encoder_weights_1600.pt'
    gmod = torch.load(gpath)

    netG = P_net()
    new_state_dict = OrderedDict()
    state_dict = gmod['state_dict']
    for k, v in state_dict.items():
        name = k  # remove `module.`
        if 'num_batches_tracked' in k:
            continue
        new_state_dict[name] = v
    netG.load_state_dict(new_state_dict)  # gmod['state_dict'])
    folder_path = './results/synthetic_images/'
    create_folder(folder_path)

    distri = torch.normal(mean=0, std=torch.ones(100, z_dim))

    for i in range(100):
        for j in range(z_dim):
            while distri[i][j] < -1 or distri[i][j] > 1:
                distri[i][j] = torch.normal(mean=0, std=torch.ones(1))[0]

    z_fake_gauss = Variable(distri)
    images = netG(z_fake_gauss)
    save_images(images, folder_path, '_synthetic')


main()
