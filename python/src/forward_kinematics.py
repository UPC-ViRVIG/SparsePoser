import torch

def fk(
    rot: torch.Tensor,
    global_pos: torch.Tensor,
    offsets: torch.Tensor,
    parents: list,
    device,
) -> torch.Tensor:
    """
    Parameters
    -----------
    rot is a tensor of shape (batch_size, n_joints * 4, frames)
    global_pos is a tensor of shape (batch_size, 3, frames)
    offsets is a tensor of shape (1, n_joints, 3)
    parents is a list of shape (n_joints)

    Returns
    --------
    result are the positions of the joints, tensor of shape: (batch_size, frames, n_joints, 3)
    """
    # change rot shape to (batch_size, n_joints, 4, frames)
    rot = rot.reshape((rot.shape[0], -1, 4, rot.shape[-1]))
    # identity quaternion
    identity = torch.tensor((1, 0, 0, 0), dtype=torch.float32, device=device)
    # change identity shape from (4,) to (1, 1, 4, 1)
    identity = identity.reshape((1, 1, -1, 1))
    new_shape = list(rot.shape)
    new_shape[1] += 1  # extra joint for easing the computation
    new_shape[2] = 1
    # repeat identity creating a tensor of shape (batch_size, n_joints + 1, 4, frames)
    # this is used to save the final rotation of each joint in each batch and frame
    # the third dimension (dim=2) starts always as identity: [1,0,0,0]
    rotation_final = identity.repeat(new_shape)
    # store current rotations shifted by one joint (e.g., root joint is rotation_final[:,1,:,:]=rotation[:,0,:,:])
    for i in range(rot.shape[1]):
        rotation_final[:, i + 1, :, :] = rot[:, i, :, :]
    # change rotation_final shape to (batch_size, fraes, n_joints, 4)
    rotation_final = rotation_final.permute(0, 3, 1, 2)
    # change global_pos shape to (batch_size, frames, n_joints, 3)
    global_pos = global_pos.permute(0, 2, 1)
    # create a result tensor of shape (batch_size, frames, n_joints, 3)
    result = torch.empty(
        rotation_final.shape[:-2] + (rotation_final.shape[-2] - 1,) + (3,),
        device=device,
    )
    # convert to unit quaternions
    norm = torch.norm(rotation_final, dim=-1, keepdim=True)
    rotation_final = rotation_final / norm
    # quaternion to transform matrix (..., 3, 3)
    transform = transform_from_quaternion(rotation_final)
    # change offset shape from (1, n_joints, 3) to (1, 1, n_joints, 3, 1) (column vector)
    offsets = offsets.reshape((-1, 1, offsets.shape[-2], offsets.shape[-1], 1))
    # first joint is global position
    result[..., 0, :] = global_pos
    # other joints are transformed by the transform matrix
    for i, parent in enumerate(parents):
        # root
        if i == 0:
            continue
        transform[..., i + 1, :, :] = torch.matmul(
            transform[..., parent + 1, :, :].clone(),
            transform[..., i + 1, :, :].clone(),
        )
        result[..., i, :] = torch.matmul(
            transform[..., parent + 1, :, :].clone(), offsets[..., i, :, :].clone()
        ).squeeze()
        result[..., i, :] = result[..., i, :] + result[..., parent, :]
    return result


def fk_2(
    rot: torch.Tensor,
    global_pos: torch.Tensor,
    offsets: torch.Tensor,
    parents: list,
    device,
) -> torch.Tensor:
    """
    Parameters
    -----------
    rot is a tensor of shape (batch_size, n_joints * 4, frames)
    global_pos is a tensor of shape (batch_size, 3, frames)
    offsets is a tensor of shape (1, n_joints, 3)
    parents is a list of shape (n_joints)

    Returns
    --------
    result are the positions of the joints, tensor of shape: (batch_size, frames, n_joints, 3)
    """
    # change rot shape to (batch_size, n_joints, 4, frames)
    rot = rot.reshape((rot.shape[0], -1, 4, rot.shape[-1]))
    # identity quaternion
    identity = torch.tensor((1, 0, 0, 0), dtype=torch.float32, device=device)
    # change identity shape from (4,) to (1, 1, 4, 1)
    identity = identity.reshape((1, 1, -1, 1))
    new_shape = list(rot.shape)
    new_shape[1] += 1  # extra joint for easing the computation
    new_shape[2] = 1
    # repeat identity creating a tensor of shape (batch_size, n_joints + 1, 4, frames)
    # this is used to save the final rotation of each joint in each batch and frame
    # the third dimension (dim=2) starts always as identity: [1,0,0,0]
    rotation_final = identity.repeat(new_shape)
    # store current rotations shifted by one joint (e.g., root joint is rotation_final[:,1,:,:]=rotation[:,0,:,:])
    for i in range(rot.shape[1]):
        rotation_final[:, i + 1, :, :] = rot[:, i, :, :]
    # change rotation_final shape to (batch_size, fraes, n_joints, 4)
    rotation_final = rotation_final.permute(0, 3, 1, 2)
    # change global_pos shape to (batch_size, frames, n_joints, 3)
    global_pos = global_pos.permute(0, 2, 1)
    # create a result tensor of shape (batch_size, frames, n_joints, 3)
    result = torch.empty(
        rotation_final.shape[:-2] + (rotation_final.shape[-2] - 1,) + (3,),
        device=device,
    )
    # convert to unit quaternions
    norm = torch.norm(rotation_final, dim=-1, keepdim=True)
    rotation_final = rotation_final / norm
    # quaternion to transform matrix (..., 3, 3)
    transform = transform_from_quaternion(rotation_final)
    # change offset shape from (1, n_joints, 3) to (1, 1, n_joints, 3, 1) (column vector)
    offsets = offsets.reshape((-1, 1, offsets.shape[-2], offsets.shape[-1], 1))
    # first joint is global position
    result[..., 0, :] = global_pos
    # other joints are transformed by the transform matrix
    for i, parent in enumerate(parents):
        # root
        if i == 0:
            continue
        transform[..., i + 1, :, :] = torch.matmul(
            transform[..., parent + 1, :, :].clone(),
            transform[..., i + 1, :, :].clone(),
        )
        result[..., i, :] = torch.matmul(
            transform[..., parent + 1, :, :].clone(), offsets[..., i, :, :].clone()
        ).squeeze()
        result[..., i, :] = result[..., i, :] + result[..., parent, :]
    return result, transform[:, :, 1:, :, :]


def transform_from_quaternion(quater: torch.Tensor):
    qw = quater[..., 0]
    qx = quater[..., 1]
    qy = quater[..., 2]
    qz = quater[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m