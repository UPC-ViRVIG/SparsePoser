import torch
import torch.nn as nn
from autoencoder import Autoencoder, StaticEncoder
from loss import MSE_DQ


class Generator_Model(nn.Module):
    def __init__(self, device, param, parents, train_data) -> None:
        super().__init__()

        self.device = device
        self.param = param
        self.parents = parents
        self.data = train_data

        self.static_encoder = StaticEncoder(param, parents, device).to(device)
        self.autoencoder = Autoencoder(param, parents, device).to(device)

        parameters = list(self.static_encoder.parameters()) + list(
            self.autoencoder.parameters()
        )

        # Print number parameters
        dec_params = 0
        for parameter in parameters:
            dec_params += parameter.numel()
        print("# parameters generator:", dec_params)

        self.optimizer = torch.optim.AdamW(parameters, param["learning_rate"])

        self.loss = MSE_DQ(param, parents, device).to(device)
        train_data.losses.append(self.loss)

    def forward(self):
        # Execute Static Encoder to obtain the offsets
        # input offsets has shape (1, n_joints, 3)
        self.ae_offsets = self.static_encoder(self.data.offsets)
        # Execute Autoencoder to obtain the motion
        self.latent = self.autoencoder(
            self.data.sparse_motion,
            self.ae_offsets,
            self.data.mean_dqs,
            self.data.std_dqs,
            self.data.denorm_offsets,
        )
        return self.res_decoder

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        loss = self.loss.forward_generator(
            self.res_decoder,
            self.data.motion,
        )
        loss.backward()
        self.optimizer.step()
        return loss
