from shutil import copyfile
from utils import *
from collections import OrderedDict
from scipy.misc import imsave
from aae import *

training_path = "./data/dataset/training_data/"
validation_path = "./data/dataset/validation_data/"
create_folder("./data/aae_results/")

encoder_path = './results/aae_models/Q_encoder_weights_1600.pt'
decoder_path = './results/aae_models/P_encoder_weights_1600.pt'
z_dim = 100
temp_path = './temp/'
create_folder(temp_path)
create_folder(temp_path + 'data')


def get_models():
    emod = torch.load(encoder_path)
    gmod = torch.load(decoder_path)
    netG = P_net()
    new_state_dict = OrderedDict()
    state_dict = gmod['state_dict']
    for k, v in state_dict.items():
        name = k  # remove `module.`
        if 'num_batches_tracked' in k:
            continue
        new_state_dict[name] = v
    netG.load_state_dict(new_state_dict)  # gmod['state_dict'])

    netE = Q_net()
    new_state_dict = OrderedDict()
    state_dict = emod['state_dict']
    for k, v in state_dict.items():
        name = k
        if 'num_batches_tracked' in k:
            continue
        new_state_dict[name] = v
    netE.load_state_dict(new_state_dict)
    return netE, netG


def save_generated_version(netE, netG, file_name):
    val_gen = dataset_iterator('./temp', batch_size=1)
    val = inf_train_gen(val_gen)
    images = next(val)
    encoding = netE(images)
    samples = netG(encoding)
    img = samples[0][0].detach().numpy()
    imsave(file_name, img)


def reconstruction_dataset(path):
    file_list = os.listdir(path + 'data/')
    folder_name = path.replace("/dataset/", "/aae_results/")
    create_folder(folder_name)
    create_folder(folder_name + "data/")

    netE, netG = get_models()
    for file_name in file_list:
        copyfile(path + 'data/' + file_name, temp_path + 'data/' + file_name)
        save_generated_version(netE, netG, folder_name + "data/" + file_name)
        os.remove(temp_path + 'data/' + file_name)


reconstruction_dataset(training_path)
reconstruction_dataset(validation_path)
