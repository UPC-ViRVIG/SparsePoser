import torch
import torch.nn as nn
from ik_net import IK_NET
from loss import MSE_DQ_FK


class IK_Model(nn.Module):
    def __init__(self, device, param, parents, train_data) -> None:
        super().__init__()

        self.device = device
        self.param = param
        self.parents = parents
        self.data = train_data

        self.ik_net = IK_NET(param, parents, device).to(device)

        # Print number parameters
        parameters = list(self.ik_net.parameters())
        dec_params = 0
        for parameter in parameters:
            dec_params += parameter.numel()
        print("# parameters ik:", dec_params)

        self.optimizer = torch.optim.AdamW(parameters, param["learning_rate"])

        self.loss = MSE_DQ_FK(param, parents, device).to(device)
        train_data.losses.append(self.loss)

    def forward(self, res_decoder, frame=None):
        self.ik_res = self.ik_net(
            res_decoder.clone().detach(),
            self.data.sparse_motion,
            self.data.mean_dqs,
            self.data.std_dqs,
            self.data.offsets,
            frame,
        )
        self.res = torch.cat([res_decoder[:, :8, :], self.ik_res], dim=1)
        return self.res

    def optimize_parameters(self, res_decoder):
        self.optimizer.zero_grad()
        loss_ik = self.loss.forward_ik(
                None,
                self.ik_res,
                res_decoder.clone().detach(),
                self.data.motion,
            )
        loss_ik.backward()
        self.optimizer.step()
        return loss_ik
