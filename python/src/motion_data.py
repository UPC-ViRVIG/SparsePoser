import numpy as np
import torch
from torch.utils.data import Dataset
import pymotion.rotations.dual_quat as dquat
from pymotion.ops.skeleton import to_root_dual_quat


class TrainMotionData(Dataset):
    def __init__(self, param, scale, device):
        self.motions = []
        self.norm_motions = []
        self.means_motions = []
        self.var_motions = []
        self.param = param
        self.scale = scale
        self.device = device

    def add_motion(self, offsets, global_pos, rotations, parents):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        global_pos: np.array of shape (n_frames, 3)
        rotations: np.array of shape (n_frames, n_joints, 4) (quaternions)
        parents: np.array of shape (n_joints)

        Returns:
        --------
        self.motions:
            offsets: tensor of shape (n_joints, 3)
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
            displacement: tensor of shape (windows_size, 3)
        """
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        # create dual quaternions
        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, parents, offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        # offsets
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        # change rotations to (1, n_joints * 4, frames)
        rots = torch.from_numpy(rotations).type(torch.float32).to(self.device)
        rots = rots.flatten(start_dim=1, end_dim=2).permute(1, 0).unsqueeze(0)
        # change global_pos to (1, 3, frames)
        global_pos = global_pos.permute(1, 0).unsqueeze(0)
        for start in range(0, frames, self.param["window_step"]):
            end = start + self.param["window_size"]
            if end < frames:
                motion = {
                    "offsets": offsets,
                    "dqs": dqs[start:end],
                    "displacement": displacement[start:end],
                }
                self.motions.append(motion)
        # Means
        motion_mean = {
            "offsets": torch.mean(offsets, dim=0).to(self.device),
            "dqs": torch.mean(dqs, dim=0).to(self.device),
            "displacement": torch.mean(displacement, dim=0).to(self.device),
        }
        self.means_motions.append(motion_mean)
        # Stds
        motion_var = {
            "offsets": torch.var(offsets, dim=0).to(self.device),
            "dqs": torch.var(dqs, dim=0).to(self.device),
            "displacement": torch.var(displacement, dim=0).to(self.device),
        }
        self.var_motions.append(motion_var)

    def normalize(self):
        """
        Normalize motions by means and stds

        Returns:
        --------
        self.norm_motions:
            offsets: tensor of shape (n_joints, 3)
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
        """
        offsets_means = torch.stack([m["offsets"] for m in self.means_motions], dim=0)
        offsets_vars = torch.stack([s["offsets"] for s in self.var_motions], dim=0)
        dqs_means = torch.stack([m["dqs"] for m in self.means_motions], dim=0)
        dqs_vars = torch.stack([s["dqs"] for s in self.var_motions], dim=0)
        displacement_means = torch.stack(
            [m["displacement"] for m in self.means_motions], dim=0
        )
        displacement_vars = torch.stack(
            [s["displacement"] for s in self.var_motions], dim=0
        )

        self.means = {
            "offsets": torch.mean(offsets_means, dim=0).to(self.device),
            "dqs": torch.mean(dqs_means, dim=0).to(self.device),
            "displacement": torch.mean(displacement_means, dim=0).to(self.device),
        }

        # Source: https://stats.stackexchange.com/a/26647
        self.stds = {
            "offsets": torch.sqrt(torch.mean(offsets_vars, dim=0)).to(self.device),
            "dqs": torch.sqrt(torch.mean(dqs_vars, dim=0)).to(self.device),
            "displacement": torch.sqrt(torch.mean(displacement_vars, dim=0)).to(
                self.device
            ),
        }

        if torch.count_nonzero(self.stds["offsets"]) != torch.numel(
            self.stds["offsets"]
        ):
            print("WARNING: offsets stds are zero")
            self.stds["offsets"][self.stds["offsets"] < 1e-10] = 1
        if torch.count_nonzero(self.stds["dqs"]) != torch.numel(self.stds["dqs"]):
            print("WARNING: dqs stds are zero")
            self.stds["dqs"][self.stds["dqs"] < 1e-10] = 1
        if torch.count_nonzero(self.stds["displacement"]) != torch.numel(
            self.stds["displacement"]
        ):
            print("WARNING: displacement stds are zero")
            self.stds["displacement"][self.stds["displacement"] < 1e-10] = 1

        # Normalized
        for motion in self.motions:
            norm_motion = {
                "offsets": (motion["offsets"] - self.means["offsets"])
                / self.stds["offsets"],
                "dqs": (motion["dqs"] - self.means["dqs"]) / self.stds["dqs"],
                "displacement": (motion["displacement"] - self.means["displacement"])
                / self.stds["displacement"],
            }
            self.norm_motions.append(norm_motion)

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        return (self.motions[index], self.norm_motions[index])


class TestMotionData:
    def __init__(self, param, scale, device):
        self.norm_motions = []
        self.bvhs = []
        self.filenames = []
        self.param = param
        self.scale = scale
        self.device = device

    def set_means_stds(self, means, stds):
        self.means = means
        self.stds = stds

    def add_motion(self, offsets, global_pos, rotations, parents, bvh, filename):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        global_pos: np.array of shape (n_frames, 3)
        rotations: np.array of shape (n_frames, n_joints, 4) (quaternions)
        parents: np.array of shape (n_joints)

        Returns:
        --------
        self.norm_motions:
            offsets: tensor of shape (n_joints, 3)
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
            displacement: tensor of shape (windows_size, 3)
        """
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        # create dual quaternions
        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, parents, offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        # offsets
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        # change rotations to (1, n_joints * 4, frames)
        rots = torch.from_numpy(rotations).type(torch.float32).to(self.device)
        rots = rots.flatten(start_dim=1, end_dim=2).permute(1, 0).unsqueeze(0)
        # change global_pos to (1, 3, frames)
        global_pos = global_pos.permute(1, 0).unsqueeze(0)
        motion = {
            "offsets": offsets,
            "denorm_offsets": offsets.clone(),
            "dqs": dqs,
            "displacement": displacement,
        }
        self.norm_motions.append(motion)
        self.bvhs.append(bvh)
        self.filenames.append(filename)

    def normalize(self):
        # Normalize
        assert self.means is not None
        assert self.stds is not None
        for motion in self.norm_motions:
            motion["offsets"] = (motion["offsets"] - self.means["offsets"]) / self.stds[
                "offsets"
            ]
            motion["dqs"] = (motion["dqs"] - self.means["dqs"]) / self.stds["dqs"]
            motion["displacement"] = (
                motion["displacement"] - self.means["displacement"]
            ) / self.stds["displacement"]

    def get_bvh(self, index):
        return self.bvhs[index], self.filenames[index]

    def get_len(self):
        return len(self.norm_motions)

    def get_item(self, index):
        return self.norm_motions[index]


class RunMotionData:
    def __init__(self, param, device):
        self.param = param
        self.device = device

    def set_means_stds(self, means, stds):
        self.means = means
        self.stds = stds

    def set_offsets(self, offsets):
        """
        Parameters:
        -----------
        offsets: np.array of shape (n_joints, 3)
        """
        offsets = torch.from_numpy(offsets).type(torch.float32).to(self.device)
        self.motion = {
            "offsets": offsets,
            "denorm_offsets": offsets.clone(),
        }

    def set_motion_from_bvh(self, offsets, global_pos, rotations, parents):
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        # create dual quaternions
        fake_global_pos = np.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, fake_global_pos, np.array(parents), offsets)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        self.motion["dqs"] = dqs
        self.motion["displacement"] = displacement

    def set_motion(self, positions, rotations):
        frames = rotations.shape[0]
        assert frames >= self.param["window_size"]
        global_pos = positions[:, 0, :]
        # create dual quaternions
        dqs = dquat.from_rotation_translation(rotations, positions)
        dqs = dquat.unroll(dqs, axis=0)  # ensure continuity
        dqs = torch.from_numpy(dqs).type(torch.float32).to(self.device)
        dqs = torch.flatten(dqs, 1, 2)
        # displacement
        global_pos = torch.from_numpy(global_pos).type(torch.float32).to(self.device)
        displacement = torch.cat(
            (torch.zeros((1, 3), device=self.device), global_pos[1:] - global_pos[:-1]),
            dim=0,
        )
        self.motion["dqs"] = dqs
        self.motion["displacement"] = displacement

    def normalize_offsets(self):
        # Normalize
        assert self.means is not None
        assert self.stds is not None
        self.motion["offsets"] = (
            self.motion["offsets"] - self.means["offsets"]
        ) / self.stds["offsets"]

    def normalize_motion(self):
        # Normalize
        assert self.means is not None
        assert self.stds is not None
        self.motion["dqs"] = (self.motion["dqs"] - self.means["dqs"]) / self.stds["dqs"]
        self.motion["displacement"] = (
            self.motion["displacement"] - self.means["displacement"]
        ) / self.stds["displacement"]

    def get_item(self):
        return self.motion
