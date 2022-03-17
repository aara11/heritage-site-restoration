import torch.nn as nn
import torch.nn.functional as F

z_dim = 100
dropout_prob = 0


# Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self._name = 'Enc'
        self.shape = (64, 64, 1)
        self.dim = z_dim
        convblock = nn.Sequential(
            nn.Conv2d(1, 10, 3, 1, padding=1, bias=True),
            nn.Conv2d(10, 10, 3, 2, padding=1, bias=True),
            nn.BatchNorm2d(10, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(10, 20, 3, 1, padding=1, bias=True),
            nn.Conv2d(20, 20, 3, 2, padding=1, bias=True),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(20, 20, 3, 1, padding=1, bias=True),
            nn.Conv2d(20, 20, 3, 2, padding=1, bias=True),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(20, 30, 3, 1, padding=1, bias=True),
            nn.Conv2d(30, 30, 3, 2, padding=1, bias=True),
            nn.BatchNorm2d(30, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(30, 30, 3, 1, padding=1, bias=True),
            nn.Conv2d(30, 30, 3, 2, padding=1, bias=True),
            nn.BatchNorm2d(30, eps=1e-05, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
        )
        self.main = convblock
        self.linear = nn.Linear(4 * 30, self.dim)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4 * 30)
        output = self.linear(output)
        return output


# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self._name = 'Gen'
        self.shape = (64, 64, 1)
        self.dim = z_dim
        preprocess1 = nn.Linear(self.dim, 4 * 30)
        preprocess2 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(30, eps=1e-05, momentum=0.9),
            nn.LeakyReLU(0.2),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(30, 30, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(30, eps=1e-05, momentum=0.9),
            nn.LeakyReLU(0.2),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(30, 30, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(30, 20, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.9),
            nn.LeakyReLU(0.2),
        )
        block3 = nn.Sequential(

            nn.ConvTranspose2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(20, eps=1e-05, momentum=0.9),
            nn.LeakyReLU(0.2),
        )
        block4 = nn.Sequential(

            nn.ConvTranspose2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(20, 10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(10, eps=1e-05, momentum=0.9),
            nn.LeakyReLU(0.2),
        )
        deconv_out = nn.Sequential(
            nn.ConvTranspose2d(10, 10, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(10, 1, kernel_size=3, stride=1, padding=1))
        self.preprocess1 = preprocess1
        self.preprocess2 = preprocess2
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.block4 = block4
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess1(input)
        output = output.view(-1, 30, 2, 2)
        output = self.preprocess2(output)
        output = self.block1(output)
        output = self.block2(output)

        output = self.block3(output)
        output = self.block4(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output


# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        return F.sigmoid(self.lin3(x))
