import torch
import train
from pymotion.ops.forward_kinematics_torch import fk


def eval_pos_error(gt_bvh, eval_bvh, device):
    gt_rots, gt_pos, gt_parents, gt_offsets, _ = train.get_info_from_bvh(gt_bvh)
    gt_rots = torch.from_numpy(gt_rots).float().to(device).unsqueeze(0)
    gt_pos = torch.zeros(gt_rots.shape[0], gt_rots.shape[1], 3)
    gt_offsets = torch.from_numpy(gt_offsets).float().to(device)
    gt_joint_poses, _ = fk(
        gt_rots, gt_pos, gt_offsets, torch.Tensor(gt_parents).long().to(device)
    )
    rots, pos, parents, offsets, _ = train.get_info_from_bvh(eval_bvh)
    # pos = (
    #     torch.from_numpy(pos).float().to(device)[:, 0, :].permute(1, 0).unsqueeze(0)
    # )  # only global position
    rots = torch.from_numpy(rots).float().to(device).unsqueeze(0)
    pos = torch.zeros(rots.shape[0], rots.shape[1], 3)
    offsets = torch.from_numpy(offsets).float().to(device)
    joint_poses, _ = fk(rots, pos, offsets, torch.Tensor(parents).long().to(device))
    # error
    error = torch.norm(
        joint_poses - gt_joint_poses[:, : joint_poses.shape[1], ...], dim=-1
    )
    sparse_error = error[:, :, train.param["sparse_joints"][1:]]  # ignore root joint
    return torch.mean(error).item(), torch.mean(sparse_error).item()
