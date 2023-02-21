import torch
import torch.nn as nn
import torch.nn.functional as F
#import ipdb; ipdb.set_trace()
# 1D conv usage:
# batch_size (N) = #3D objects , channels = features, signal_lengt (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution over only all features of one point sample


# 3D Single View Reconsturction (for 256**3 input voxelization) --------------------------------------
# ----------------------------------------------------------------------------------------------------

class NDF(nn.Module):


    def __init__(self, hidden_dim=256):
        super(NDF, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='border')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='border')  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='border')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='border')  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='border')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='border')  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 8

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()
        self.latent_dim = 1741
        #self.displacments = torch.Tensor(displacments).cuda()
        self.num_codebook_vectors = 2048*3
        self.beta = 1
        self.embedding_1 = nn.Embedding(self.num_codebook_vectors, self.latent_dim).cuda()
        self.embedding_1.weight.data.uniform_(0, 1).cuda()


    def encoder(self,x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6

    def embedding(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):
        # pe_embd = self.harmonic_embedding(p).transpose(1, -1)

        p_features = p.transpose(1, -1)

        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # feature extraction
        feature_0 = F.grid_sample(f_0, p, align_corners=True)
        feature_1 = F.grid_sample(f_1.float(), p, align_corners=True)
        feature_2 = F.grid_sample(f_2.float(), p, align_corners=True)
        feature_3 = F.grid_sample(f_3.float(), p, align_corners=True)
        feature_4 = F.grid_sample(f_4.float(), p, align_corners=True)
        feature_5 = F.grid_sample(f_5.float(), p, align_corners=True)
        feature_6 = F.grid_sample(f_6.float(), p, align_corners=True)

        # here every channel corresponds to one feature.

        features = torch.cat(
            (feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6), dim=1
        )  # (B, features, 1,7,sample_num)
        shape = features.shape
        #print(f" The shape of features is {shape}")
        features = torch.reshape(
            features, (shape[0], shape[1] * shape[3], shape[4])
        )  # (B, featues_per_sample, samples_num)
        # features = torch.cat((features, p_features, pe_embd), dim=1)  # (B, featue_size, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)
        
        z = features
        #print(f"The shape of z is: {z.shape}")
        z_flattened = z.contiguous().view(-1, self.latent_dim)

        #print(f"The shape of flattened z_flattened is : {z_flattened.shape}")
        #print(self.embedding_1)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True)+ \
            torch.sum(self.embedding_1.weight ** 2, dim=1) - \
            2 * (torch.matmul(z_flattened,self.embedding_1.weight.t()))
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding_1(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach() - z) **2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q -z).detach()
        
        return z_q, min_encoding_indices, loss


    def decoder(self, p, z_q):

        net = self.actvn(self.fc_0(z_q))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)

        return  out

    def forward(self, p, x):
        z_q, min_encoding_indices, loss = self.embedding(p, *self.encoder(x))
        out = self.decoder(p, z_q)
        return out, min_encoding_indices, loss



#T = torch.rand(3,256,256,256).cuda()
#p = torch.rand(3,3000,3).cuda()
#N = NDF().cuda()
#e_1,e_2, e_3, e_4, e_5, e_6, e_7 = N.encoder(T)
#print(e_1,e_2, e_3, e_4, e_5, e_6, e_7)
#z_q, min_encoding_indices, loss = N.embedding(p, e_1,e_2, e_3, e_4, e_5, e_6, e_7)
#print(z_q)
#out = N.decoder(p, z_q)
#print(out)
