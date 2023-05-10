import quat
import numpy as np
import torch


def skeleton_to_dual_quat(offsets, rotations, parents):
    """
    Convert the skeleton information to dual quaternions.

    Parameters
    ----------
    offsets : np.array[n_joints, 3]
        The offset of the joint from its parent.
    rotations : np.array[frames, n_joints, 4]
        The rotation of the joint.
    global_pos: np.array[frames, 3]
        The global position of the root joint.
    parents : [n_joints]
        The parent of the joint.

    Returns
    -------
    dual_quat : np.array[frames, n_joints, 8]
        The dual quaternion representation of the skeleton.
    """
    assert (offsets[0] == np.zeros(3)).all()
    n_frames = rotations.shape[0]
    n_joints = rotations.shape[1]
    # translations has shape (frames, n_joints, 3)
    rotations = rotations.copy()
    translations = np.tile(offsets, (n_frames, 1, 1))
    # make transformations local to the root
    for j in range(1, n_joints):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        translations[:, j, :] = (
            quat.mul_vec(rotations[:, parent, :], translations[:, j, :])
            + translations[:, parent, :]
        )
        rotations[:, j, :] = quat.mul(rotations[:, parent, :], rotations[:, j, :])
    # convert to dual quaternions
    dual_quat = rot_trans_to_dual_quat(rotations, translations)
    return dual_quat


def skeleton_from_dual_quat(dq, parents):
    """
    Convert the dual quaternion to the skeleton information.

    Parameters
    ----------
    dq : np.array[frames, n_joints, 8]
        Includes as first element the global position of the root joint
    parents : [n_joints]

    Returns
    -------
    rotations : np.array[frames, n_joints, 4]
    translations : np.array[frames, n_joints, 3]
    """
    n_joints = dq.shape[1]
    # rotations has shape (frames, n_joints, 4)
    # translations has shape (frames, n_joints, 3)
    rotations, translations = rot_trans_from_dual_quat(dq)
    # make transformations local to the parents
    # (initially local to the root)
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        translations[:, j, :] = quat.mul_vec(
            quat.inv(rotations[:, parent, :]),
            translations[:, j, :] - translations[:, parent, :],
        )
        rotations[:, j, :] = quat.mul(
            quat.inv(rotations[:, parent, :]), rotations[:, j, :]
        )
    return rotations, translations


def skeleton_from_dual_quat_py(dq, parents, device):
    # PYTORCH IMPLEMENTATION
    """
    Convert the dual quaternion to the skeleton information.

    Parameters
    ----------
    dq : np.array[..., frames, n_joints, 8]
        Includes as first element the global position of the root joint
    parents : [n_joints]

    Returns
    -------
    rotations : np.array[..., frames, n_joints, 4]
    translations : np.array[..., frames, n_joints, 3]
    """
    n_joints = dq.shape[-2]
    # rotations has shape (..., frames, n_joints, 4)
    # translations has shape (..., frames, n_joints, 3)
    rotations, translations = rot_trans_from_dual_quat_py(dq, device)
    # make transformations local to the parents
    # (initially local to the root)
    for j in reversed(range(1, n_joints)):
        parent = parents[j]
        if parent == 0:  # already in root space
            continue
        inv = quat.inv_py(rotations[..., :, parent, :].clone())
        translations[..., :, j, :] = quat.mul_vec_py(
            inv,
            translations[..., :, j, :] - translations[..., :, parent, :],
        ).clone()
        rotations[..., :, j, :] = quat.mul_py(inv, rotations[..., :, j, :].clone())
    return rotations, translations


def rot_trans_from_dual_quat_py(dq, device):
    """
    PYTORCH VERSION
    Convert a dual quaternion to the rotations and translations

    Parameters
    ----------
    dq: np.array[..., 8]

    Returns
    -------
    rotations: np.array[..., 4]
    translations: np.array[..., 3]
    """
    # DEBUG
    assert is_unit_py(dq, device)
    dq = dq.clone()
    q_r = dq[..., :4]
    # rotations can ge get directly from the real part of the dual quaternion
    rotations = q_r
    q_d = dq[..., 4:]
    # the translation (pure quaternion) t = 2 * q_d * q_r*
    # where q_r* is the conjugate of q_r
    translations = (2 * quat.mul_py(q_d.clone(), quat.conjugate_py(q_r).clone()))[
        ..., 1:
    ]
    return rotations, translations


def rot_trans_from_dual_quat(dq):
    """
    Convert a dual quaternion to the rotations and translations

    Parameters
    ----------
    dq: np.array[..., 8]

    Returns
    -------
    rotations: np.array[..., 4]
    translations: np.array[..., 3]
    """
    # DEBUG
    assert is_unit(dq)
    dq = dq.copy()
    q_r = dq[..., :4]
    # rotations can ge get directly from the real part of the dual quaternion
    rotations = q_r
    q_d = dq[..., 4:]
    # the translation (pure quaternion) t = 2 * q_d * q_r*
    # where q_r* is the conjugate of q_r
    translations = (2 * quat.mul(q_d, quat.conjugate(q_r)))[..., 1:]
    return rotations, translations


def rot_trans_to_dual_quat(rotations, translations):
    """
    Convert the rotation and translation information to dual quaternions.

    Parameters
    ----------
    rotations : np.array[..., 4]
    translations : np.array[..., 3]

    Returns
    -------
    dual_quat : np.array[..., 8]
    """
    # dual quaternion (sigma) = qr + qd (+ is not an addition, but concatenation)
    # real part of dual quaternions represent rotations
    # and is represented as a conventional unit quaternion
    q_r = rotations
    # dual part of dual quaternions represent translations
    # t is a pure quaternion (0, x, y, z)
    # q_d = 0.5 * eps * t * q_r
    t = np.zeros((translations.shape[:-1] + (4,)))
    t[..., 1:] = translations
    q_d = 0.5 * quat.mul(t, q_r)
    dq = np.concatenate((q_r, q_d), axis=-1)
    # DEBUG
    assert is_unit(dq)
    return dq


def rot_trans_to_dual_quat_py(rotations, translations, device):
    # PYTORCH IMPLEMENTATION
    """
    Convert the rotation and translation information to dual quaternions.

    Parameters
    ----------
    rotations : np.array[..., 4]
    translations : np.array[..., 3]

    Returns
    -------
    dual_quat : np.array[..., 8]
    """
    # dual quaternion (sigma) = qr + qd (+ is not an addition, but concatenation)
    # real part of dual quaternions represent rotations
    # and is represented as a conventional unit quaternion
    q_r = rotations
    # dual part of dual quaternions represent translations
    # t is a pure quaternion (0, x, y, z)
    # q_d = 0.5 * eps * t * q_r
    t = torch.zeros((translations.shape[:-1] + (4,))).to(device)
    t[..., 1:] = translations
    q_d = 0.5 * quat.mul_py(t, q_r)
    dq = torch.cat((q_r, q_d), dim=-1)
    # DEBUG
    assert is_unit_py(dq, device)
    return dq


def dual_quaternion_from_translation(translations):
    """
    Convert a translation to a dual quaternion.

    Parameters
    ----------
    translations : np.array[..., 3]

    Returns
    -------
    dual_quats : np.array[..., 8]
    """
    dual_quats = np.zeros((translations.shape[:-1] + (8,)))
    dual_quats[..., 0:1] = 1
    dual_quats[..., 5:] = translations / 2
    return dual_quats


def normalize_py(dq, device):
    """
    Normalize the dual quaternion.

    Parameters
    ----------
    dq: np.array[..., 8]

    Returns
    -------
    dq: np.array[..., 8]
    """
    dq = dq.clone()
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    norm = torch.norm(q_r, dim=-1)
    qnorm = torch.stack((norm, norm, norm, norm), dim=-1)
    q_r_normalized = torch.div(q_r, qnorm)
    q_d_normalized = torch.div(q_d, qnorm)
    if not is_unit_py(torch.cat((q_r_normalized, q_d_normalized), dim=-1), device):
        # make sure that the dual quaternion is orthogonal to the real quaternion
        dot_q_r_q_d = torch.sum(q_r * q_d, dim=-1)  # dot product of q_r and q_d
        q_d_normalized_ortho = q_d_normalized - (
            q_r_normalized * torch.div(dot_q_r_q_d, norm * norm).unsqueeze(-1)
        )
        dq = torch.cat((q_r_normalized, q_d_normalized_ortho), dim=-1)
    else:
        dq = torch.cat((q_r_normalized, q_d_normalized), dim=-1)
    assert is_unit_py(dq, device)  # DEBUG
    return dq


def is_unit_py(dq, device, atol=1e-03):
    # pytorch implementation
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    sqr_norm_q_r = torch.sum(q_r * q_r, dim=-1)
    if torch.isclose(
        sqr_norm_q_r, torch.zeros(sqr_norm_q_r.shape, device=device)
    ).all():
        return True
    rot_normalized = torch.isclose(
        sqr_norm_q_r, torch.ones(sqr_norm_q_r.shape, device=device)
    ).all()
    sqr_norm_q_d = torch.sum(q_r * q_d, dim=-1)
    trans_normalized = torch.isclose(
        sqr_norm_q_d, torch.zeros(sqr_norm_q_d.shape, device=device), atol=atol
    ).all()
    return rot_normalized and trans_normalized


def is_unit(dq, atol=1e-03):
    """
    Check if the dual quaternion is a unit one

    Parameters
    ----------
    dq: np.array[..., 8]
    """
    q_r = dq[..., :4]
    q_d = dq[..., 4:]
    sqr_norm_q_r = np.sum(q_r * q_r, axis=-1)
    if np.isclose(sqr_norm_q_r, 0).all():
        return True
    rot_normalized = np.isclose(sqr_norm_q_r, 1).all()
    trans_normalized = np.isclose(np.sum(q_r * q_d, axis=-1), 0, atol=atol).all()
    return rot_normalized and trans_normalized


def homo_mat_to_dual_quat(mat):
    """
    Convert a homogeneous matrix (4x4) to a dual quaternion

    Parameters
    ----------
    mat: np.array[..., 4, 4]

    Returns
    -------
    dq: np.array[..., 8]
    """
    raise NotImplementedError


def unroll(dq):
    """
    Enforce dual quaternion continuity across the time dimension by selecting
    the representation (dq or -dq) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.

    Parameters
    ----------
    dq : np.array[time, ..., 8]

    Returns
    -------
    dq : np.array[time, ..., 8]
    """
    dq = dq.copy()
    q_r = dq[..., :4]
    # start with the second quaternion since
    # we keep the cover of the first one
    for i in range(1, len(q_r)):
        # distance (dot product) between the previous and current quaternion
        d0 = np.sum(q_r[i] * q_r[i - 1], axis=-1)
        # distance (dot product) between the previous and flipped current quaternion
        d1 = np.sum(-q_r[i] * q_r[i - 1], axis=-1)
        # if the distance with the flipped quaternion is smaller, use it
        dq[i][d0 < d1] = -dq[i][d0 < d1]
    return dq
