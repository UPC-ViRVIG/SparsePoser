import numpy as np
import torch


def from_scaled_angle_axis(axisangle):
    """
    Create a quaternion from an angle-axis representation.

    Parameters
    ----------
    angle : np.array[..., angle]
    axis : np.array[..., [x,y,z]]
        normalized axis [x,y,z] of rotation

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    angle = np.linalg.norm(axisangle, axis=-1)
    axis = axisangle / angle[..., np.newaxis]
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    return np.concatenate((c, s * axis), axis=-1)


def from_angle_axis(angle, axis):
    """
    Create a quaternion from an angle-axis representation.

    Parameters
    ----------
    angle : np.array[..., angle]
    axis : np.array[..., [x,y,z]]
        normalized axis [x,y,z] of rotation

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    return np.concatenate((c, s * axis), axis=-1)


def from_euler(euler, order):
    """
    Create a quaternion from an euler representation with a specified order.

    Parameters
    ----------
    euler : np.array[..., 3]
    order : np.array[..., 3]
        order of the euler angles, last three components are strings in ['x', 'y', 'z']

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    axis = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }
    q0 = from_angle_axis(euler[..., 0], np.array([axis[o] for o in order[..., 0]]))
    q1 = from_angle_axis(euler[..., 1], np.array([axis[o] for o in order[..., 1]]))
    q2 = from_angle_axis(euler[..., 2], np.array([axis[o] for o in order[..., 2]]))
    return mul(q0, mul(q1, q2))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def from_matrix(rotations):
    """
    Convert rotation matrices to quaternions.
    Args:
        rotations: as tensor of shape (..., 9). Matrix order: (c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z) where ci is column i.
    Returns:
        Quaternions (w, x, y, z) as tensor of shape (..., 4).
    """
    # Separate components
    r00, r10, r20, r01, r11, r21, r02, r12, r22 = torch.unbind(rotations, -1)
    # Quaternion
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + r00 + r11 + r22,
                1.0 + r00 - r11 - r22,
                1.0 - r00 + r11 - r22,
                1.0 - r00 - r11 + r22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, r21 - r12, r02 - r20, r10 - r01], dim=-1),
            torch.stack([r21 - r12, q_abs[..., 1] ** 2, r10 + r01, r02 + r20], dim=-1),
            torch.stack([r02 - r20, r10 + r01, q_abs[..., 2] ** 2, r12 + r21], dim=-1),
            torch.stack([r10 - r01, r20 + r02, r21 + r12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates


def to_euler(quaternions, order):
    """
    Convert a quaternion to an euler representation with a specified order.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]
    order : np.array[..., 3]
        order of the euler angles, last three components are strings in ['x', 'y', 'z']

    Returns
    -------
    euler : np.array[..., 3]
    """
    aux = {
        "x": "X",
        "y": "Y",
        "z": "Z",
    }
    return matrix_to_euler_angles(
        to_matrix(quaternions), aux[order[0, 0]] + aux[order[0, 1]] + aux[order[0, 2]]
    )


def to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Parameters
    ----------
        quaternions: np.array[..., [w,x,y,z]]
    Returns
    -------
        rotation_matrices: np.array[..., 3, 3]
    """
    w = quaternions[..., 0]
    x = quaternions[..., 1]
    y = quaternions[..., 2]
    z = quaternions[..., 3]
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = np.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_euler_angles(matrices, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.
    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrices.shape[-1] != 3 or matrices.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrices.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = np.arcsin(
            matrices[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = np.arccos(matrices[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrices[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrices[..., i0, :], True, tait_bryan
        ),
    )
    return np.stack(o, -1)


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.
    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.
    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return np.arctan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return np.arctan2(-data[..., i2], data[..., i1])
    return np.arctan2(data[..., i2], -data[..., i1])


def _fast_cross(a, b):
    """
    Fast cross of two vectors

    Parameters
    ----------
    a : np.array[..., [x,y,z]]
    b : np.array[..., [x,y,z]]

    Returns
    -------
    np.array[..., [x,y,z]]
    """

    return np.concatenate(
        [
            a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
            a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
            a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1],
        ],
        axis=-1,
    )


def mul_vec(q, v):
    """
    Multiply a vector by a quaternion

    Parameters
    ----------
    q : np.array[..., [w,x,y,z]]
    v : np.array[..., [x,y,z]]

    Returns
    -------
    v: np.array[..., [x,y,z]]
    """
    t = 2.0 * _fast_cross(q[..., 1:], v)
    return v + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)


def _fast_cross_py(a, b):
    # PYTROCH version
    return torch.cat(
        [
            a[..., 1:2] * b[..., 2:3] - a[..., 2:3] * b[..., 1:2],
            a[..., 2:3] * b[..., 0:1] - a[..., 0:1] * b[..., 2:3],
            a[..., 0:1] * b[..., 1:2] - a[..., 1:2] * b[..., 0:1],
        ],
        dim=-1,
    )


def mul_vec_py(q, v):
    # PYTORCH Version
    t = 2.0 * _fast_cross_py(q[..., 1:], v)
    return v + q[..., 0:1] * t + _fast_cross_py(q[..., 1:], t)


def mul(q0, q1):
    """
    Multiply two quaternions.

    Parameters
    ----------
    q0 : np.array[..., [w,x,y,z]]
    q1 : np.array[..., [w,x,y,z]]

    Returns
    -------
    quat : np.array[..., [w,x,y,z]]
    """
    w0, x0, y0, z0 = q0[..., 0:1], q0[..., 1:2], q0[..., 2:3], q0[..., 3:4]
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    # (w0,v0)(w1,v1) = (w0w1 - v0·v1, w0v1 + w1v0 + v0 x v1)
    return np.concatenate(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,  # w
            w0 * x1 + w1 * x0 + y0 * z1 - z0 * y1,  # x
            w0 * y1 + w1 * y0 + z0 * x1 - x0 * z1,  # y
            w0 * z1 + w1 * z0 + x0 * y1 - y0 * x1,  # z
        ),
        axis=-1,
    )


def mul_py(q0, q1):
    """
    PYTORCH VERSION
    Multiply two quaternions.

    Parameters
    ----------
    q0 : tensor [..., [w,x,y,z]]
    q1 : tensor [..., [w,x,y,z]]

    Returns
    -------
    quat : tensor [..., [w,x,y,z]]
    """
    w0, x0, y0, z0 = torch.unbind(q0, -1)
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    # (w0,v0)(w1,v1) = (w0w1 - v0·v1, w0v1 + w1v0 + v0 x v1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,  # w
            w0 * x1 + w1 * x0 + y0 * z1 - z0 * y1,  # x
            w0 * y1 + w1 * y0 + z0 * x1 - x0 * z1,  # y
            w0 * z1 + w1 * z0 + x0 * y1 - y0 * x1,  # z
        ),
        dim=-1,
    )


def length(quaternions):
    """
    Get the length or magnitude of the quaternions.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    length : np.array[...]
    """
    return np.sqrt(np.sum(quaternions * quaternions, axis=-1))


def length_py(quaternions):
    # PYTORCH
    return torch.sqrt(torch.sum(quaternions * quaternions, dim=-1))


def normalize(quaternions, eps=1e-8):
    """
    Convert all quaternions to unit quatenrions.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    return quaternions / (length(quaternions)[..., np.newaxis] + eps)


def normalize_py(quaternions, eps=1e-8):
    # PYTORCH
    return quaternions / (length_py(quaternions).unsqueeze(-1) + eps)


def to_axisangle(quaternions):
    """
    Quaternion to scaled axis angle representation.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    axisangle : np.array[..., [x,y,z]]
    """
    q = normalize(quaternions)
    angle = 2 * np.arccos(q[..., 0:1])
    s = np.sqrt(1 - q[..., 0:1] * q[..., 0:1])
    return angle * q[..., 1:] / s


def inv(quaternions):
    """
    Inverse of a quaternion.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    # for a unit quaternion the conjugate is the inverse
    # q^-1 = [q0, -q1, -q2, -q3]
    return conjugate(quaternions)


def inv_py(quaternions):
    # PYTORCH IMPLEMENTATION
    """
    Inverse of a quaternion.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    # for a unit quaternion the conjugate is the inverse
    # q^-1 = [q0, -q1, -q2, -q3]
    return conjugate_py(quaternions)


def unroll(quaternions):
    """
    Avoid the quaternion 'double cover' problem by picking the cover
    of the first quaternion, and then removing sudden switches
    over the cover by ensuring that each frame uses the quaternion
    closest to the one of the previous frame.

    ('double cover': same rotation can be encoded with two
    different quaternions)

    Usage example: Ensuring an animation to have quaternions
    that represent the 'shortest' rotation path. Otherwise,
    if we SLERP between poses we would get joints rotating in
    the "longest" path.

    Parameters
    ----------
    quaternions : np.array[time, [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[time, [w,x,y,z]]
    """
    r = quaternions.copy()
    # start with the second quaternion since
    # we keep the cover of the first one
    for i in range(1, len(quaternions)):
        # distance (dot product) between the previous and current quaternion
        d0 = np.sum(r[i] * r[i - 1], axis=-1)
        # distance (dot product) between the previous and flipped current quaternion
        d1 = np.sum(-r[i] * r[i - 1], axis=-1)
        # if the distance with the flipped quaternion is smaller (or, equivalently, maximal dot product), use it
        r[i][d0 < d1] = -r[i][
            d0 < d1
        ]  # unit quaternion (angle, v) represents same rotation as (-angle, -v)
    return r


def conjugate(quaternions):
    """
    Compute the conjugate of a quaternion.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    return np.concatenate((quaternions[..., 0:1], -quaternions[..., 1:]), axis=-1)


def conjugate_py(quaternions):
    """
    Compute the conjugate of a quaternion.

    Parameters
    ----------
    quaternions : np.array[..., [w,x,y,z]]

    Returns
    -------
    quaternions : np.array[..., [w,x,y,z]]
    """
    return torch.cat((quaternions[..., 0:1], -quaternions[..., 1:]), dim=-1)
