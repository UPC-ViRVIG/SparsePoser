import torch
import torch.nn as nn
from pymotion.ops.forward_kinematics_torch import fk
from pymotion.ops.skeleton_torch import from_root_dual_quat


class MSE_DQ(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.param = param
        self.parents = parents
        self.device = device

    def set_mean(self, mean_dqs):
        self.mean_dqs = mean_dqs.unsqueeze(-1)

    def set_std(self, std_dqs):
        self.std_dqs = std_dqs.unsqueeze(-1)

    def set_offsets(self, offsets):
        self.joint_distances = torch.norm(offsets, dim=1)

    def forward_generator(self, input, target):
        dqs = input.clone()
        target_dqs = target.clone()
        # Dual Quaternions MSE Loss
        loss_joints = self.mse(dqs[:, 8:, :], target_dqs[:, 8:, :])
        loss_root = self.mse(dqs[:, :8, :], target_dqs[:, :8, :])
        return loss_root * self.param["lambda_root"] + loss_joints


class MSE_DQ_FK(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.param = param
        self.parents = torch.Tensor(parents).long().to(device)
        self.device = device

        # indices without sparse input
        self.indices_no_sparse = []
        for i in range(0, 22):
            if i not in self.param["sparse_joints"]:
                self.indices_no_sparse.append(i)
        self.indices_no_sparse = torch.tensor(self.indices_no_sparse).to(self.device)

    def set_mean(self, mean_dqs):
        self.mean_dqs = mean_dqs.unsqueeze(-1)

    def set_std(self, std_dqs):
        self.std_dqs = std_dqs.unsqueeze(-1)

    def set_offsets(self, offsets):
        self.offsets = offsets  # denormalized
        self.joint_distances = torch.norm(offsets, dim=1)

    def forward_ik(self, input, input_ik, input_decoder, target):
        dqs = input_ik.clone()
        dqs_decoder = input_decoder.clone()
        target_dqs = target.clone()
        # add fake root rotation to input_ik (it is to keep the code compatible with previous versions)
        # without using the root rotation predicted from the decoder
        dqs = torch.cat((torch.zeros_like(dqs[:, :8, :]), dqs), dim=1)
        # FK Loss
        # denormalize dual quaternions
        dqs = dqs * self.std_dqs + self.mean_dqs
        dqs[:, :8, :] = (
            torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).unsqueeze(0).unsqueeze(-1)
        )  # make root quaternion identity (so asserts will pass)
        dqs = dqs.reshape((dqs.shape[0], -1, 8, dqs.shape[-1])).permute(0, 3, 1, 2)
        dqs_decoder = dqs_decoder * self.std_dqs + self.mean_dqs
        dqs_decoder[:, :8, :] = (
            torch.tensor([1, 0, 0, 0, 0, 0, 0, 0]).unsqueeze(0).unsqueeze(-1)
        )  # make root quaternion identity (no root)
        dqs_decoder = dqs_decoder.reshape(
            (dqs_decoder.shape[0], -1, 8, dqs_decoder.shape[-1])
        ).permute(0, 3, 1, 2)
        target_dqs = target_dqs * self.std_dqs + self.mean_dqs
        target_dqs = target_dqs.reshape(
            (target_dqs.shape[0], -1, 8, target_dqs.shape[-1])
        ).permute(0, 3, 1, 2)
        # compute rotations and translations local to their parents
        _, local_rot = from_root_dual_quat(dqs, self.parents)
        local_rot[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        _, local_rot_decoder = from_root_dual_quat(dqs_decoder, self.parents)
        local_rot_decoder[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        _, target_local_rot = from_root_dual_quat(target_dqs, self.parents)
        target_local_rot[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        # compute final positions (root space) using standard FK
        joint_poses, joint_rot_mat = fk(
            local_rot,
            torch.zeros((local_rot.shape[0], local_rot.shape[1], 3)),
            self.offsets,
            self.parents,
        )
        target_joint_poses, target_joint_rot_mat = fk(
            target_local_rot,
            torch.zeros((target_local_rot.shape[0], target_local_rot.shape[1], 3)),
            self.offsets,
            self.parents,
        )
        decoder_joint_poses, decoder_rot_mat = fk(
            local_rot_decoder,
            torch.zeros((local_rot_decoder.shape[0], local_rot_decoder.shape[1], 3)),
            self.offsets,
            self.parents,
        )

        # positions
        loss_ee = self.mse(
            joint_poses[:, :, self.param["sparse_joints"][1:], :],
            target_joint_poses[:, :, self.param["sparse_joints"][1:], :],
        )
        # rotations
        loss_ee += self.mse(
            joint_rot_mat[:, :, self.param["sparse_joints"][1:], :, :],
            target_joint_rot_mat[:, :, self.param["sparse_joints"][1:], :, :],
        )

        # regularization
        loss_ee_reg = self.mse(
            decoder_joint_poses[:, :, self.indices_no_sparse, :],
            joint_poses[:, :, self.indices_no_sparse, :],
        )
        loss_ee_reg += self.mse(
            decoder_rot_mat[:, :, self.indices_no_sparse, :, :],
            joint_rot_mat[:, :, self.indices_no_sparse, :, :],
        )

        return (
            loss_ee * self.param["lambda_ee"]
            + loss_ee_reg * self.param["lambda_ee_reg"]
        )
