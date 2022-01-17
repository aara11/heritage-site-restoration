from scipy.misc import imsave, imread
from aae import *
import numpy as np
from torch.autograd import Variable
from utils import *


def generate_ae_image(iter, netE, netG, save_path, args, real_data):
    batch_size = args.batch_size
    datashape = netE.shape
    encoding = netE(real_data)
    samples = netG(encoding)
    if netG._name == 'mnistG':
        samples = samples.view(batch_size, 28, 28)
    else:
        samples = samples.view(-1, *(datashape[::-1]))
        samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(samples, save_path + '/ae_samples_{}.jpg'.format(iter))


def discriminatorAccuracy(netD, netE, netG):
    distri = torch.normal(mean=0, std=torch.ones(1000, z_dim))
    for i in range(1000):
        for j in range(z_dim):
            while distri[i][j] < -1 or distri[i][j] > 1:
                distri[i][j] = torch.normal(mean=0, std=torch.ones(1))[0]

    z_fake_gauss = Variable(distri)
    D_fake_gauss = netD(netE(netG(z_fake_gauss)))
    val = D_fake_gauss
    val = [1.0 - x for x in val]
    val = np.log(val)
    # print(len([x for x in val if x >= -1]))
    print(max(val))
    print(min(val))

    val_gen = dataset_iterator("./data/dataset/validation_data/")
    val = inf_train_gen(val_gen)
    images = next(val)
    D_real_gauss = netD(netE(images))
    val = D_real_gauss
    val = sorted(val)
    print(max(val))
    print(min(val))


def saveAll_imagesDecoded(netE, netG):
    folder_ = './data/training_data/data/'
    file_names = os.listdir(folder_)
    save_path = './aae_results/val_results'
    create_folder(save_path)

    input_ = []
    for name in file_names:
        path = folder_ + name
        image = imread(path)
        print(image.shape)
        img = image.reshape([64, 64, 1])
        input_.append(img)
        imsave(save_path + name, image)
    input_ = torch.tensor(np.array(input_))
    images_gen = netG(netE(input_))
    for i in range(images_gen.shape[0]):
        img = images_gen[i][0].detach().numpy()
        imsave(save_path + name[:-4] + '_decoded.jpg', img)


def main1():
    gpath = './results/aae_models/P_encoder_weights_120.pt'
    epath = './results/aae_models/Q_encoder_weights_120.pt'
    dpath = './results/aae_models/D_encoder_weights_120.pt'
    netE, netG, netD = read_aae_weights(epath=epath, gpath=gpath, dpath=dpath)
    discriminatorAccuracy(netD, netE, netG)

    # saveAll_imagesDecoded(netE, netG)


batch_size = 64
z_dim = 100

main1()
