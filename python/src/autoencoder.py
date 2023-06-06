import torch
import torch.nn as nn
import pymotion.rotations.dual_quat_torch as dquat
from skeleton import (
    SkeletonPool,
    SkeletonUnpool,
    find_neighbor,
    SkeletonConv,
    SkeletonLinear,
    create_pooling_list,
)


class Autoencoder(nn.Module):
    def __init__(self, param, parents, device):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(param, parents, device)
        self.decoder = Decoder(param, self.encoder, device)
        self.parents = parents

    def forward(self, input, offset, mean_dqs, std_dqs, denorm_offsets):
        latent = self.encoder(input)
        output = self.decoder(
            latent, offset, mean_dqs, std_dqs, denorm_offsets, self.parents
        )
        return latent, output


class Encoder(nn.Module):
    def __init__(self, param, parents, device):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        self.convs = []
        self.parents = [parents]
        self.pooling_lists = []
        self.channel_list = []

        kernel_size = param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2
        stride = param["stride_encoder_conv"]

        # Compute pooled skeletons
        number_layers = 3
        layer_parents = parents
        for l in range(number_layers):
            pooling_list, layer_parents = create_pooling_list(
                layer_parents, l == number_layers - 1
            )
            self.pooling_lists.append(pooling_list)
            self.parents.append(layer_parents)

        default_channels_per_joints = 8
        self.channel_size = [default_channels_per_joints] + [
            (default_channels_per_joints) * (2**i)
            for i in range(1, number_layers + 1)
        ]  # first each joint has 8 (dual quaternion) channels, then we multiply by 2 every layer to increase the number of conv. filters
        low_res_parents = self.parents[-1]
        neighbor_list, _ = find_neighbor(
            low_res_parents, param["neighbor_distance"], add_displacement=True
        )
        self.num_joints = len(neighbor_list)  # it includes the displacement fake joint
        for l in range(number_layers):
            seq = []
            in_channels = self.channel_size[l] * self.num_joints
            out_channels = self.channel_size[l + 1] * self.num_joints
            if l == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channel_size[l],
                    out_channels_per_joint=self.channel_size[l + 1],
                    joint_num=self.num_joints,
                    in_offset_channel=3 * self.channel_size[-1] // self.channel_size[0],
                    padding=padding,
                    stride=stride,
                    device=device,
                    add_offset=False,
                )
            )
            self.convs.append(seq[-1])
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            # Append to the list of layers
            self.layers.append(nn.Sequential(*seq))

    def forward(self, input):
        # if using dual quaternions, all channels have 8 components but the last one (displacement)
        # has 3 channels. Pad the last one with zeros so that all have 8 components
        input = torch.cat(
            (input, torch.zeros_like(input[:, [0, 1, 2, 3, 4], :])), dim=1
        )
        # forward
        for layer in self.layers:
            input = layer(input)
        return input


class Decoder(nn.Module):
    def __init__(self, param, enc: Encoder, device):
        super(Decoder, self).__init__()

        self.param = param
        self.device = device
        self.layers = nn.ModuleList()
        self.convs = []

        kernel_size = param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2

        number_layers = 3
        default_channels_per_joints = 8
        self.channel_size = [default_channels_per_joints] + [
            default_channels_per_joints * (2**i) for i in range(1, number_layers + 1)
        ]  # first each joint has 8 channels, after collapse 16, then 32...
        for i in range(number_layers):
            seq = []

            neighbor_list, _ = find_neighbor(
                enc.parents[number_layers - i - 1], param["neighbor_distance"]
            )
            num_joints = len(neighbor_list)

            unpool = SkeletonUnpool(
                pooling_list=enc.pooling_lists[number_layers - i - 1],
                channels_per_edge=self.channel_size[number_layers - i],
                device=device,
            )
            seq.append(nn.Upsample(scale_factor=2, mode="linear", align_corners=False))
            seq.append(unpool)

            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=self.channel_size[number_layers - i],
                    out_channels_per_joint=self.channel_size[number_layers - i] // 2,
                    joint_num=num_joints,
                    in_offset_channel=3
                    * enc.channel_size[number_layers - i - 1]
                    // enc.channel_size[0],
                    padding=padding,
                    stride=1,
                    device=device,
                    add_offset=True,
                )
            )
            self.convs.append(seq[-1])
            if i != number_layers - 1:
                seq.append(nn.LeakyReLU(negative_slope=0.2))
            # Append to the list of layers
            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset, mean_dqs, std_dqs, denorm_offsets, parents):
        for i, layer in enumerate(self.layers):
            self.convs[i].set_offset(offset[len(self.layers) - i - 1])
            input = layer(input)
        # input has shape (batch_size, num_joints * 8, frames)
        # denormalize rotations
        input = input * std_dqs.unsqueeze(-1) + mean_dqs.unsqueeze(-1)
        # change input to shape (batch_size, frames, num_joints, 8)
        input = input.reshape(input.shape[0], -1, 8, input.shape[-1]).permute(
            0, 3, 1, 2
        )
        # convert to unit dual quaternions
        input = dquat.normalize(input)
        # normalize rotations
        input = input.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        input = (input - mean_dqs.unsqueeze(-1)) / std_dqs.unsqueeze(-1)
        return input


# encoder for static part, i.e. offset part
class StaticEncoder(nn.Module):
    def __init__(self, param, parents, device):
        super(StaticEncoder, self).__init__()
        self.layers = nn.ModuleList()
        channels = 3  # position

        number_layers = 3
        layer_parents = parents
        for i in range(number_layers):
            neighbor_list, _ = find_neighbor(layer_parents, param["neighbor_distance"])
            seq = []
            seq.append(
                SkeletonLinear(
                    param=param,
                    neighbor_list=neighbor_list,
                    in_channels=channels * len(neighbor_list),
                    out_channels=channels * 2 * len(neighbor_list),
                    device=device,
                )
            )
            if i < number_layers - 1:
                pool = SkeletonPool(
                    parents=layer_parents, channels_per_edge=channels * 2, device=device
                )
                layer_parents = pool.new_parents
                seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            channels *= 2
            self.layers.append(nn.Sequential(*seq))

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        output = [input]
        for layer in self.layers:
            input = layer(input)
            input = input.squeeze(-1)
            output.append(input)
        return output
