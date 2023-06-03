import torch
import torch.nn as nn
import pymotion.rotations.dual_quat_torch as dquat


class IK_NET(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()

        self.param = param
        self.parents = parents
        self.device = device

        channels_per_joint = 8
        channels_per_offset = 3

        # offsets leg: LowerLeg, Foot, Toe
        # offsets arm: UpperArm, LowerArm, Hand
        # dq leg: UpperLeg, LowerLeg, Foot
        # dq arm: Shoulder, UpperArm, LowerArm
        # Left Leg ----------------------------------------
        self.l_leg_offsets = [2, 3, 4]
        l_leg_dq_in = [3]
        while l_leg_dq_in[-1] != 0:
            l_leg_dq_in.append(self.parents[l_leg_dq_in[-1]])
        self.l_leg_dq_in_extended = [
            i
            for j in l_leg_dq_in
            for i in range(
                j * channels_per_joint, j * channels_per_joint + channels_per_joint
            )
        ]
        l_leg_dq_out = [1, 2, 3]
        self.l_leg_dq_out_extended = [
            i
            for j in l_leg_dq_out
            for i in range(
                j * channels_per_joint, j * channels_per_joint + channels_per_joint
            )
        ]
        # Right Leg ----------------------------------------
        self.r_leg_offsets = [6, 7, 8]
        r_leg_dq_in = [7]
        while r_leg_dq_in[-1] != 0:
            r_leg_dq_in.append(self.parents[r_leg_dq_in[-1]])
        self.r_leg_dq_in_extended = [
            i
            for j in r_leg_dq_in
            for i in range(
                j * channels_per_joint, j * channels_per_joint + channels_per_joint
            )
        ]
        r_leg_dq_out = [5, 6, 7]
        self.r_leg_dq_out_extended = [
            i
            for j in r_leg_dq_out
            for i in range(
                j * channels_per_joint, j * channels_per_joint + channels_per_joint
            )
        ]
        # Left Arm ----------------------------------------
        self.l_arm_offsets = [15, 16, 17]
        l_arm_dq_in = [16]
        while l_arm_dq_in[-1] != 0:
            l_arm_dq_in.append(self.parents[l_arm_dq_in[-1]])
        self.l_arm_dq_in_extended = [
            i
            for j in l_arm_dq_in
            for i in range(
                j * channels_per_joint, j * channels_per_joint + channels_per_joint
            )
        ]
        l_arm_dq_out = [14, 15, 16]
        self.l_arm_dq_out_extended = [
            i
            for j in l_arm_dq_out
            for i in range(
                j * channels_per_joint, j * channels_per_joint + channels_per_joint
            )
        ]
        # Right Arm ----------------------------------------
        self.r_arm_offsets = [19, 20, 21]
        r_arm_dq_in = [20]
        while r_arm_dq_in[-1] != 0:
            r_arm_dq_in.append(self.parents[r_arm_dq_in[-1]])
        self.r_arm_dq_in_extended = [
            i
            for j in r_arm_dq_in
            for i in range(
                j * channels_per_joint, j * channels_per_joint + channels_per_joint
            )
        ]
        r_arm_dq_out = [18, 19, 20]
        self.r_arm_dq_out_extended = [
            i
            for j in r_arm_dq_out
            for i in range(
                j * channels_per_joint, j * channels_per_joint + channels_per_joint
            )
        ]

        # Network ----------------------------------------
        hidden_size = 128
        self.sequential_l_leg = nn.Sequential(
            nn.Linear(
                len(self.l_leg_offsets) * channels_per_offset  # offsets
                + len(self.l_leg_dq_in_extended)  # input pose
                + channels_per_joint,  # sparse input (end effector)
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(
                hidden_size, len(self.l_leg_dq_out_extended)
            ),  # output modified joints
        ).to(device)
        self.sequential_r_leg = nn.Sequential(
            nn.Linear(
                len(self.r_leg_offsets) * channels_per_offset  # offsets
                + len(self.r_leg_dq_in_extended)  # input pose
                + channels_per_joint,  # sparse input (end effector)
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(
                hidden_size, len(self.r_leg_dq_out_extended)
            ),  # output modified joints
        ).to(device)
        self.sequential_l_arm = nn.Sequential(
            nn.Linear(
                len(self.l_arm_offsets) * channels_per_offset  # offsets
                + len(self.l_arm_dq_in_extended)  # input pose
                + channels_per_joint,  # sparse input (end effector)
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(
                hidden_size, len(self.l_arm_dq_out_extended)
            ),  # output modified joints
        ).to(device)
        self.sequential_r_arm = nn.Sequential(
            nn.Linear(
                len(self.r_arm_offsets) * channels_per_offset  # offsets
                + len(self.r_arm_dq_in_extended)  # input pose
                + channels_per_joint,  # sparse input (end effector)
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(
                hidden_size, len(self.r_arm_dq_out_extended)
            ),  # output modified joints
        ).to(device)

    def forward(
        self,
        decoder_output,
        sparse_input,
        mean_dqs,
        std_dqs,
        offsets,
        frame,
    ):
        # decoder_output shape: (batch_size, n_joints*8, frames)
        # sparse_input shape: (batch_size, n_sparse_joints*8, frames)
        sparse_input = sparse_input[:, 8:, :]  # exclude root
        sparse_input = sparse_input[:, :-3, :]  # exclude displacement
        n_frames = decoder_output.shape[-1]
        res_l_leg = self.forward_limb(
            self.sequential_l_leg,
            n_frames,
            sparse_input[:, [0, 1, 2, 3, 4, 5, 6, 7], :],
            offsets,
            decoder_output,
            self.l_leg_offsets,
            self.l_leg_dq_in_extended,
            self.l_leg_dq_out_extended,
            std_dqs,
            mean_dqs,
            frame,
        )
        res_r_leg = self.forward_limb(
            self.sequential_r_leg,
            n_frames,
            sparse_input[:, [8, 9, 10, 11, 12, 13, 14, 15], :],
            offsets,
            decoder_output,
            self.r_leg_offsets,
            self.r_leg_dq_in_extended,
            self.r_leg_dq_out_extended,
            std_dqs,
            mean_dqs,
            frame,
        )
        res_l_arm = self.forward_limb(
            self.sequential_l_arm,
            n_frames,
            sparse_input[:, [24, 25, 26, 27, 28, 29, 30, 31], :],
            offsets,
            decoder_output,
            self.l_arm_offsets,
            self.l_arm_dq_in_extended,
            self.l_arm_dq_out_extended,
            std_dqs,
            mean_dqs,
            frame,
        )
        res_r_arm = self.forward_limb(
            self.sequential_r_arm,
            n_frames,
            sparse_input[:, [32, 33, 34, 35, 36, 37, 38, 39], :],
            offsets,
            decoder_output,
            self.r_arm_offsets,
            self.r_arm_dq_in_extended,
            self.r_arm_dq_out_extended,
            std_dqs,
            mean_dqs,
            frame,
        )
        # combine with decoder_output
        if frame is None:
            decoder_output[:, self.l_leg_dq_out_extended, :] = res_l_leg
            decoder_output[:, self.r_leg_dq_out_extended, :] = res_r_leg
            decoder_output[:, self.l_arm_dq_out_extended, :] = res_l_arm
            decoder_output[:, self.r_arm_dq_out_extended, :] = res_r_arm
        else:
            decoder_output[:, self.l_leg_dq_out_extended, frame] = res_l_leg.squeeze(-1)
            decoder_output[:, self.r_leg_dq_out_extended, frame] = res_r_leg.squeeze(-1)
            decoder_output[:, self.l_arm_dq_out_extended, frame] = res_l_arm.squeeze(-1)
            decoder_output[:, self.r_arm_dq_out_extended, frame] = res_r_arm.squeeze(-1)
        return decoder_output[:, 8:, :]

    def forward_limb(
        self,
        sequential,
        n_frames,
        sparse_input,
        offsets,
        decoder_output,
        offsets_i,
        dq_in_extended,
        dq_out_extended,
        std_dqs,
        mean_dqs,
        frame_i,
    ):
        if frame_i is None:
            res = sequential(
                torch.cat(
                    (
                        torch.tile(
                            offsets[:, offsets_i, :]
                            .flatten(start_dim=1, end_dim=2)
                            .unsqueeze(-1),
                            (1, 1, n_frames),
                        ),
                        decoder_output[:, dq_in_extended, :],
                        sparse_input,  # end effectors
                    ),
                    dim=-2,
                ).permute(0, 2, 1)
            ).permute(0, 2, 1)
        else:
            res = sequential(
                torch.cat(
                    (
                        offsets[:, offsets_i, :].flatten(start_dim=1, end_dim=2),
                        decoder_output[:, dq_in_extended, frame_i],
                        sparse_input[:, :, frame_i],  # end effector
                    ),
                    dim=-1,
                )
            ).unsqueeze(-1)
        # denormalize rotations
        res = res * std_dqs[dq_out_extended].unsqueeze(-1) + mean_dqs[
            dq_out_extended
        ].unsqueeze(-1)
        # change res to shape (batch_size, frames, num_joints, 8)
        res = res.reshape(res.shape[0], -1, 8, res.shape[-1]).permute(0, 3, 1, 2)
        # convert to unit dual quaternions
        res = dquat.normalize(res)
        # normalize rotations
        res = res.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        res = (res - mean_dqs[dq_out_extended].unsqueeze(-1)) / std_dqs[
            dq_out_extended
        ].unsqueeze(-1)
        return res
