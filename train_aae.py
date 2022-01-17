from aae import *
from utils import *
from scipy.misc import imsave
from torch.autograd import Variable

logs = {}
z_dim = 100
recon_loss_lst = []
G_loss_lst = []
D_loss_lst = []
val_recon_loss_lst = []

EPS = 1e-15
Q = Q_net()  # .cuda()
P = P_net()  # .cuda()
D_gauss = D_net_gauss()  # .cuda()
batch_size = 64

# Set learning rates
gen_lr = 0.002
reg_lr = 0.0002

create_folder("./results/")
create_folder("./results/aae_results/")
create_folder("./results/aae_models/")
# encode/decode optimizers
optim_P = torch.optim.Adam(P.parameters(), lr=gen_lr)
optim_Q_enc = torch.optim.Adam(Q.parameters(), lr=gen_lr)
# regularizing optimizers
optim_Q_gen = torch.optim.Adam(Q.parameters(), lr=reg_lr)
optim_D = torch.optim.Adam(D_gauss.parameters(), lr=reg_lr)

train_gen = dataset_iterator('./data/dataset/training_data')
dev_gen = dataset_iterator('./data/dataset/training_data')
val_gen = dataset_iterator('./data/dataset/validation_data/', batch_size=50)
gen = inf_train_gen(train_gen)
dev = inf_train_gen(dev_gen)
val = inf_train_gen(val_gen)

total_step = 500000

# Start training
for stp in range(total_step):

    images = next(gen)
    # reconstruction loss
    P.zero_grad()
    Q.zero_grad()
    D_gauss.zero_grad()

    z_sample = Q(images)  # encode to z
    X_sample = P(z_sample)  # decode to X reconstruction
    def_rec_loss = nn.MSELoss()
    p = X_sample[0][0]
    recon_loss = def_rec_loss(X_sample, images)
    recon_loss.backward()
    optim_P.step()
    optim_Q_enc.step()

    # Discriminator
    ## true prior is random normal (randn)
    ## this is constraining the Z-projection to be normal!
    Q.eval()
    distri = torch.normal(mean=0, std=torch.ones(images.size()[0], z_dim))
    for i in range(images.size()[0]):
        for j in range(z_dim):
            while distri[i][j] < -1 or distri[i][j] > 1:
                distri[i][j] = torch.normal(mean=0, std=torch.ones(1))[0]

    z_real_gauss = Variable(distri)

    D_real_gauss = D_gauss(z_real_gauss)
    z_fake_gauss = Q(images)
    D_fake_gauss = D_gauss(z_fake_gauss)
    D_loss = -torch.mean(torch.log(D_real_gauss + EPS) + torch.log(1 - D_fake_gauss + EPS))
    D_loss.backward()
    optim_D.step()

    # Generator
    Q.train()
    distri = torch.normal(mean=0, std=torch.ones(images.size()[0], z_dim))
    for i in range(images.size()[0]):
        for j in range(z_dim):
            while distri[i][j] < -1 or distri[i][j] > 1:
                distri[i][j] = torch.normal(mean=0, std=torch.ones(1))[0]

    z_fake_gauss = Variable(distri)
    D_fake_gauss = D_gauss(z_fake_gauss)

    G_loss = -torch.mean(torch.log(D_fake_gauss + EPS))

    G_loss.backward()
    optim_Q_gen.step()

    recon_loss_lst.append(recon_loss.data.numpy())
    G_loss_lst.append(G_loss.data.numpy())
    D_loss_lst.append(D_loss.data.numpy())
    info = {
        'recon_loss': recon_loss.data,
        'discriminator_loss': D_loss.data,
        'generator_loss': G_loss.data
    }
    print(info)
    if (stp + 1) % 20 == 0:
        print(stp + 1)
        save_path = './results/aae_results/' + str(stp + 1) + '/'
        create_folder(save_path)

        images = next(dev)
        for i in range(len(images)):
            img = images[i][0].detach().numpy()
            imsave(save_path + str(i) + '.jpg', img)

        images_gen = P(Q(images))
        for i in range(len(images_gen)):
            img = images_gen[i][0].detach().numpy()
            imsave(save_path + str(i) + '_decoded.jpg', img)

        images = next(val)
        P.zero_grad()
        Q.zero_grad()

        images_gen = P(Q(images))  # decode to X reconstruction
        def_rec_loss = nn.MSELoss()
        recon_loss = def_rec_loss(images_gen, images)

        print(recon_loss)
        for i in range(len(images)):
            img = images[i][0].detach().numpy()
            imsave(save_path + 'val_' + str(i) + '.jpg', img)

        for i in range(len(images_gen)):
            img = images_gen[i][0].detach().numpy()
            imsave(save_path + 'val_' + str(i) + '_decoded.jpg', img)

        val_recon_loss_lst.append(recon_loss.data.numpy())
        logs['recon_loss'] = recon_loss_lst
        logs['discriminator_loss'] = G_loss_lst
        logs['generator_loss'] = D_loss_lst
        logs['validation loss'] = val_recon_loss_lst
        save_pickle('log.pkl', logs)

        # save the weights
        P_state = {'epoch': stp + 1, 'state_dict': P.state_dict(), 'optimizer': optim_P.state_dict()}
        Q_state = {'epoch': stp + 1, 'state_dict': Q.state_dict(), 'optimizer_enc': optim_Q_enc.state_dict(),
                   'optimizer_gen': optim_Q_gen.state_dict()}
        D_state = {'epoch': stp + 1, 'state_dict': D_gauss.state_dict(), 'optimizer': optim_D.state_dict()}
        torch.save(P_state, './results/aae_models/P_encoder_weights_' + str(stp + 1) + '.pt')
        torch.save(Q_state, './results/aae_models/Q_encoder_weights_' + str(stp + 1) + '.pt')
        torch.save(D_state, './results/aae_models/D_encoder_weights_' + str(stp + 1) + '.pt')
