import torch
import torch.nn as nn
from forward_kinematics import fk_2
import dual_quat as dquat

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
        return (
            loss_root * self.param["lambda_root"]
            + loss_joints
        )

class MSE_DQ_FK(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.param = param
        self.parents = parents
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
        local_rot, _ = dquat.skeleton_from_dual_quat_py(dqs, self.parents, self.device)
        local_rot[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        local_rot = local_rot.flatten(start_dim=2, end_dim=3).permute(0, 2, 1)
        local_rot_decoder, _ = dquat.skeleton_from_dual_quat_py(
            dqs_decoder, self.parents, self.device
        )
        local_rot_decoder[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        local_rot_decoder = local_rot_decoder.flatten(start_dim=2, end_dim=3).permute(
            0, 2, 1
        )
        target_local_rot, _ = dquat.skeleton_from_dual_quat_py(
            target_dqs, self.parents, self.device
        )
        target_local_rot[..., 0, :] = torch.tensor([1, 0, 0, 0]).to(
            self.device
        )  # no global rotation
        target_local_rot = target_local_rot.flatten(start_dim=2, end_dim=3).permute(
            0, 2, 1
        )
        # compute final positions (root space) using standard FK
        joint_poses, joint_rot_mat = fk_2(
            local_rot,
            torch.zeros((local_rot.shape[0], 3, local_rot.shape[2])),
            self.offsets.unsqueeze(0),
            self.parents,
            self.device,
        )
        target_joint_poses, target_joint_rot_mat = fk_2(
            target_local_rot,
            torch.zeros((target_local_rot.shape[0], 3, target_local_rot.shape[2])),
            self.offsets.unsqueeze(0),
            self.parents,
            self.device,
        )
        decoder_joint_poses, decoder_rot_mat = fk_2(
            local_rot_decoder,
            torch.zeros((local_rot_decoder.shape[0], 3, local_rot_decoder.shape[2])),
            self.offsets.unsqueeze(0),
            self.parents,
            self.device,
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
        # loss_ee = self.mse(
        #     joint_poses[:, :, self.param["sparse_joints"][1:], :],
        #     target_joint_poses[:, :, self.param["sparse_joints"][1:], :],
        # )

        # loss_ee_reg = self.mse(input_decoder[:, 8:, :], input_ik)
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
