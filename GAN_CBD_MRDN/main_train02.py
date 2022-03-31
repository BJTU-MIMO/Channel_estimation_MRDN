import scipy.io as sio
import numpy as np
import torch.nn as nn
import torch
import torchvision
import torch.utils.data as data
from torch.autograd import Variable
import os

import math
from scipy.fftpack import fft2, ifft2, fft, ifft
import math

from GAN_model import G_CBDNet
from GAN_model import D_discriminater_Net
from function_forme import compute_SNR, fft_reshape, fft_shrink, add_noise_improve, real_imag_stack, Noise_map, \
    tensor_reshape, ifft_tensor, compute_NMSE, test, nomal_noiselevel


os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDE_VISIBLE_DIVICES"] = "0"

EPOCH = 300
BATCH_SIZE = 80
TEST_SIZE = 20
LR = 0.001
focus = 0.1
img_height = 64
img_width = 32
img_channels = 1
Max_abs = 2500

Step = 0
mat = sio.loadmat('data/train.mat')
x_train = mat['H_ori']

x, y, H, H_get = fft_reshape(x_train, img_height, img_width)
print(np.shape(H_get))       # 4000 64 32
train_loader = data.DataLoader(dataset=H_get, batch_size=BATCH_SIZE, shuffle=True)
G_net = G_CBDNet(num_input_channels=1)
D_net = D_discriminater_Net(num_input_channels=1)
device_ids = [0]

G = nn.DataParallel(G_net, device_ids=device_ids).cuda()
D = nn.DataParallel(D_net, device_ids=device_ids).cuda()

Loss = nn.MSELoss()
Loss.cuda()

criterion = nn.BCELoss() 
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(EPOCH):
    tr_loss = 0.0
    running_loss = 0.0
    for i, x in enumerate(train_loader, 0):

        NN = len(x)

        real_label = Variable(torch.ones(NN)).cuda()          
        fake_label = Variable(torch.zeros(NN)).cuda()  
        sx = x.numpy()

        H_train_data = np.zeros([NN, img_height, img_width * 2], dtype=complex)    # 64 64
        H_train_data = H_train_data + sx

        H = fft_shrink(H_train_data, img_height, img_width)                        # 64 32

        noise, E_output1 = add_noise_improve(H, 20, 20.1)

        H_n = noise + H
        SNR = compute_SNR(H, noise)
        n_level = nomal_noiselevel(E_output1)

        # fft2
        H_n_fft = np.zeros([BATCH_SIZE, 64, 32], dtype=complex)
        H_fft = np.zeros([BATCH_SIZE, 64, 32], dtype=complex)
        for i_num in range(BATCH_SIZE):
            H_n_fft[i_num, :, :] = fft2(H_n[i_num, :, :])
            H_fft[i_num, :, :] = fft2(H[i_num, :, :])

        # 64x32->64x64 numpy real+imag
        H_n_fft_r_i = real_imag_stack(H_n_fft)
        noise_r_i = H_n_fft_r_i - real_imag_stack(H_fft)
        H_n_fft_stack = tensor_reshape(H_n_fft_r_i)
        noise_stack = tensor_reshape(noise_r_i)

        n_level_01 = torch.from_numpy(n_level)
        H_n_fft_train = Variable(H_n_fft_stack.cuda())
        real_img = Variable(noise_stack.cuda())

        for i_num in range(BATCH_SIZE):
            real_img[i_num, :, :] = real_img[i_num, :, :]/(100*n_level[i_num])

        H_n_fft_fake = 100 * H_n_fft_train / Max_abs + 120
        real_out = D(real_img) 
        d_loss_real = criterion(real_out, real_label) 
        real_scores = real_out 

        # z = Variable(torch.randn(num_img, z_dimension)).cuda()  
        fake_img = G(H_n_fft_fake) 
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label) 
        fake_scores = fake_out 

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(H_n_fft_fake)
        # g_loss = Loss(fake_img, real_img / focus)
        label_real_d = D(fake_img)
        g_loss = criterion(label_real_d, real_label)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        h_fft_pre = torch.zeros([BATCH_SIZE, 64, 64])

        for i_num in range(BATCH_SIZE):
            h_fft_pre[i_num, :, :] = H_n_fft_train[i_num, :, :] - fake_img[i_num, :, :] * focus * (100*n_level_01[i_num])

        ssx = h_fft_pre.detach().numpy()
        ssx_i = np.zeros([NN, 64, 64], dtype=complex)  # 64 X 64
        ssx_i = ssx + ssx_i
        H_fft_pre_last = fft_shrink(ssx_i, img_height, img_width)
        H_fft_pre_last1, H_fft_pre_last2 = ifft_tensor(H_fft_pre_last)

        NMSE = compute_NMSE(H, H_fft_pre_last1)
        if i % 20 == 19:
            tr_loss = running_loss / 20

        if Step % 2 == 0:
            # G.apply(svd_orthogonalization)
            print("[epoch %d][%d/%d] g_loss: %.4f d_loss: %.4f SNR: %.4f NMSE: %.4f" % (
            epoch + 1, i + 1, len(train_loader), g_loss, d_loss, SNR, NMSE))
        Step += 1


