import numpy as np
import time
import torch
from PIL import Image
from scipy.misc import imsave
import os
from utils import *
from PixelRnn import *


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform):
        self.file_name = os.listdir(folder_path.replace("/dataset/", "/aae_results/") + "data")
        self.transform = transform
        self.root_dir = folder_path

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir.replace("/dataset/", "/aae_results/") + 'data', self.file_name[idx])
        aae_generated = Image.open(img_name)
        aae_generated = self.transform(aae_generated)
        img_name = os.path.join(self.root_dir + 'data/', self.file_name[idx])
        original = Image.open(img_name)
        original = self.transform(original)
        sample = {'aae_generated': aae_generated, 'original': original, 'file_name': self.file_name[idx]}
        return sample


def loss_val(rnn_created, original, loss_fn):
    actual = (original + 1) / 2.0 * 255.0
    actual = actual.type(torch.LongTensor).reshape(actual.shape[0], actual.shape[2], -1)
    loss = loss_fn(rnn_created, actual)
    return loss


def save_images(save_path, tensor, file_name, type_='class', pre="", sub=""):
    for i in range(len(file_name)):
        if type_ == 'class':
            img = tensor[i].detach().numpy()
            img = np.argmax(img, axis=1)
        elif type_ == 'value':
            img = tensor[i][0].detach().numpy()
        else:
            print("error")
        imsave(save_path + pre + file_name[i][:-4] + sub + file_name[i][-4:], img)


def test(network, input_tensor):
    img = torch.zeros(batch_size, 1, 64, 64)
    rnn_output = torch.zeros(batch_size, 256, 64, 64)
    for i in range(64):
        for j in range(64):
            output = network(input_tensor, img)
            rnn_output[:, :, i, j] = output[:, :, i, j]
            output = output.detach().numpy()
            output = np.argmax(output, axis=1)
            output = ((output / 255.0) - 0.5) * 2
            output = torch.unsqueeze(torch.tensor(output), 1)
            img[:, :, i, j] = output[:, :, i, j]
    return img, rnn_output


training_path = "./data/dataset/training_data/"
validation_path = "./data/dataset/validation_data/"
hidden_dim = 40
lr = 0.005
num_layers = 7
batch_size = 16
logs = {}
train_loss = []
validaton_loss = []

num_epoch = 22

train_transform = get_transform()
TrainSet = TrainingDataset(training_path, train_transform)
TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=batch_size, shuffle=True, num_workers=0)

validation_transform = get_transform()
ValSet = TrainingDataset(validation_path, validation_transform)
ValLoader = torch.utils.data.DataLoader(ValSet, batch_size=batch_size, shuffle=True, num_workers=0)

network = RowLSTM((64, 64), 1)
loss_fn = nn.CrossEntropyLoss()
optim_network = torch.optim.RMSprop(network.parameters(), lr=lr)
create_folder('./results/pixelrnn_model/')
create_folder('./results/pixelrnn_results/')

for epoch in range(num_epoch):
    val = []
    start = time.time()
    for i_batch, sample_batched in enumerate(TrainLoader, 0):
        optim_network.zero_grad()
        aae_generated = sample_batched['aae_generated']
        if aae_generated.shape[0] < batch_size:
            continue
        original = sample_batched['original']
        rnn_created = network(aae_generated, original)
        loss = loss_val(rnn_created, original, loss_fn)
        val.append(loss.data.numpy())
        loss.backward()
        optim_network.step()

    if epoch + 1 == num_epoch:
        save_path = './results/pixelrnn_results/' + str(epoch + 1) + '/'
        create_folder(save_path)
        # loss = []
        # for i_batch, sample_batched in enumerate(ValLoader, 0):
        #     aae_generated = sample_batched['aae_generated']
        #     if aae_generated.shape[0] < batch_size:
        #         continue
        #     original = sample_batched['original']
        #     loss_fn = nn.CrossEntropyLoss()
        #     img, rnn_output = test(network, aae_generated)
        #
        #     loss.append(loss_val(rnn_output, original, loss_fn).data.numpy())
        #     save_images(save_path, img, sample_batched['file_name'], 'value', 'val', 'rnn')
        #     save_images(save_path, original, sample_batched['file_name'], 'value', 'val', '')
        #     save_images(save_path, aae_generated, sample_batched['file_name'], 'value', 'val', 'aae')
        # validaton_loss.append(np.mean(loss))
        #
        # logs['validation loss'] = validaton_loss
        # save_pickle('log.pkl', logs)
    state = {'epoch': epoch + 1, 'state_dict': network.state_dict(), 'optimizer': optim_network.state_dict()}
    torch.save(state, './results/pixelrnn_model/' + str(epoch + 1) + '.pt')
