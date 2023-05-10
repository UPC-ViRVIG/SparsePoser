import torch
import train
from forward_kinematics import fk

def eval_pos_error(gt_bvh, eval_bvh, device):
    gt_rots, gt_pos, gt_parents, gt_offsets, _ = train.get_info_from_bvh(gt_bvh)
    # gt_pos = (
    #     torch.from_numpy(gt_pos).float().to(device)[:, 0, :].permute(1, 0).unsqueeze(0)
    # )  # only global position
    gt_rots = (
        torch.from_numpy(gt_rots)
        .float()
        .to(device)
        .flatten(start_dim=1, end_dim=2)
        .permute(1, 0)
        .unsqueeze(0)
    )
    gt_pos = torch.zeros(gt_rots.shape[0], 3, gt_rots.shape[2])
    gt_offsets = torch.from_numpy(gt_offsets).float().to(device).unsqueeze(0)
    gt_joint_poses = fk(gt_rots, gt_pos, gt_offsets, gt_parents, device)
    rots, pos, parents, offsets, _ = train.get_info_from_bvh(eval_bvh)
    # pos = (
    #     torch.from_numpy(pos).float().to(device)[:, 0, :].permute(1, 0).unsqueeze(0)
    # )  # only global position
    rots = (
        torch.from_numpy(rots)
        .float()
        .to(device)
        .flatten(start_dim=1, end_dim=2)
        .permute(1, 0)
        .unsqueeze(0)
    )
    pos = torch.zeros(rots.shape[0], 3, rots.shape[2])
    offsets = torch.from_numpy(offsets).float().to(device).unsqueeze(0)
    joint_poses = fk(rots, pos, offsets, parents, device)
    # error
    error = torch.norm(
        joint_poses - gt_joint_poses[:, : joint_poses.shape[1], ...], dim=-1
    )
    sparse_error = error[:, :, train.param["sparse_joints"][1:]]  # ignore root joint
    return torch.mean(error).item(), torch.mean(sparse_error).item()
