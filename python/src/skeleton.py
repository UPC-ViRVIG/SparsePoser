import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SkeletonConv(nn.Module):
    def __init__(
        self,
        param,
        neighbor_list,
        kernel_size,
        in_channels_per_joint,
        out_channels_per_joint,
        joint_num,
        in_offset_channel,
        padding,
        stride,
        device,
        add_offset=True,
    ):
        super().__init__()

        self.param = param
        self.add_offset = add_offset
        self.device = device
        self.padding = padding
        self.stride = stride
        self._padding_repeated_twice = (padding, padding)
        self.padding_mode = "reflect" 

        in_channels = in_channels_per_joint * joint_num
        out_channels = out_channels_per_joint * joint_num
        self.in_channels_per_joint = in_channels_per_joint
        self.out_channels_per_joint = out_channels_per_joint
        if in_channels % joint_num != 0 or out_channels % joint_num != 0:
            raise Exception(
                "in_channels and out_channels must be divisible by joint_num"
            )

        # expanded points use channels instead of joints
        self.expanded_neighbor_list = create_expanded_neighbor_list(
            neighbor_list, self.in_channels_per_joint
        )

        # use a separated matrix multiplication for the offsets
        # (less computation needed than doing convolution with rotations + offsets)
        self.offset_enc = SkeletonLinear(
            param,
            neighbor_list,
            in_offset_channel * joint_num,
            out_channels,
            device,
        )

        # weight is a matrix of size (out_channels, in_channels, kernel_size) containing the
        # convolution parameters learned from the data in a temporal window of kernel_size
        self.weight = torch.zeros(out_channels, in_channels, kernel_size).to(
            self.device
        )
        self.bias = torch.zeros(out_channels).to(self.device)
        # mask is a matrix of size (out_channels, in_channels, kernel_size) containing
        # which channels of the input affect the output (repeated in the dim=2)
        self.mask = torch.zeros_like(self.weight).to(self.device)

        self.description = (
            "SkeletonConv(in_channels_per_joint={}, out_channels_per_joint={}, kernel_size={}, "
            "joint_num={}, stride={}, padding={})".format(
                in_channels_per_joint,
                out_channels_per_joint,
                kernel_size,
                joint_num,
                stride,
                padding,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbor in enumerate(self.expanded_neighbor_list):
            """Use temporary variable to avoid assign to copy of slice, which might lead to un expected result"""
            tmp = torch.zeros_like(
                self.weight[
                    self.out_channels_per_joint
                    * i : self.out_channels_per_joint
                    * (i + 1),
                    neighbor,
                    ...,
                ]
            )
            self.mask[
                self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1),
                neighbor,
                ...,
            ] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[
                self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1),
                neighbor,
                ...,
            ] = tmp
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight[
                    self.out_channels_per_joint
                    * i : self.out_channels_per_joint
                    * (i + 1),
                    neighbor,
                    ...,
                ]
            )
            bound = 1 / math.sqrt(fan_in)
            tmp = torch.zeros_like(
                self.bias[
                    self.out_channels_per_joint
                    * i : self.out_channels_per_joint
                    * (i + 1)
                ]
            )
            nn.init.uniform_(tmp, -bound, bound)
            self.bias[
                self.out_channels_per_joint * i : self.out_channels_per_joint * (i + 1)
            ] = tmp

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.bias = nn.Parameter(self.bias)

    def set_offset(self, offset):
        self.offset = offset.reshape(offset.shape[0], -1)

    def forward(self, input):
        # pytorch is channel first:
        # weights = (out_channels, in_channels, kernel1D_size)
        weight_masked = self.weight * self.mask
        res = F.conv1d(
            F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
            weight_masked,
            self.bias,
            self.stride,
            padding=0,
            dilation=1,
            groups=1,
        )

        if self.add_offset:
            # https://github.com/DeepMotionEditing/deep-motion-editing/issues/89
            offset_res = self.offset_enc(self.offset)
            res += offset_res
        return res


class SkeletonLinear(nn.Module):
    def __init__(
        self,
        param,
        neighbor_list,
        in_channels,
        out_channels,
        device,
    ):
        super().__init__()

        self.param = param
        self.device = device

        joint_num = len(neighbor_list)
        # out_channels is alaways twice in_channels because every time we pool
        # each new joint represents two old joints
        self.in_channels_per_joint = in_channels // joint_num
        self.out_channels_per_joint = out_channels // joint_num
        if (
            in_channels % joint_num != 0
            or out_channels % joint_num != 0
        ):
            raise Exception(
                "in_channels and out_channels must be divisible by joint_num"
            )

        # expanded points use channels instead of joints
        self.expanded_neighbor_list = create_expanded_neighbor_list(
            neighbor_list, self.in_channels_per_joint
        )

        # weight is a matrix of size (out_channels, in_channels) containing the
        # convolution parameters learned from the data
        self.weight = torch.zeros(out_channels, in_channels).to(self.device)
        # mask is a matrix of size (out_channels, in_channels) containing
        # which channels of the input affect the output
        self.mask = torch.zeros_like(self.weight).to(self.device)
        self.bias = nn.Parameter(torch.Tensor(out_channels).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        for i, neighbor in enumerate(self.expanded_neighbor_list):
            tmp = torch.zeros_like(
                self.weight[
                    i
                    * self.out_channels_per_joint : (i + 1)
                    * self.out_channels_per_joint,
                    neighbor,
                ]
            )
            self.mask[
                i * self.out_channels_per_joint : (i + 1) * self.out_channels_per_joint,
                neighbor,
            ] = 1
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[
                i * self.out_channels_per_joint : (i + 1) * self.out_channels_per_joint,
                neighbor,
            ] = tmp

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        weight_masked = self.weight * self.mask
        output = F.linear(input, weight_masked, self.bias)
        output = output.reshape(output.shape + (1,))
        return output


def create_pooling_list(parents, add_displacement=False):
    # vertices degree
    degrees, direct_neighbors = create_degree_list(parents)

    # collapse_joints contains the indices of joints that will be collapsed
    collapse_joints = find_collapse_joints(degrees, direct_neighbors)
    # self.pooling_list contains merged joints, the i-th element is the
    # i-th new joint after pooling, and the indices in the list are the collapsed joints
    pooling_list = []
    old_to_new = {}  # old_to_new[old_joint] = new_joint
    new_to_old = {}  # new_to_old[new_joint] = old_joint
    for old_j in range(len(parents)):
        if old_j not in collapse_joints:
            # new joint
            new_j = len(pooling_list)
            pooling_list.append([old_j])
            old_to_new[old_j] = new_j
            new_to_old[new_j] = old_j
    # add collapsed joints to new joints lists
    for old_j in range(len(parents)):
        if old_j in collapse_joints:
            for neighbor in direct_neighbors[old_j]:
                if neighbor != old_j and neighbor != len(
                    parents
                ):  # not itself or displacement
                    pooling_list[old_to_new[neighbor]].append(old_j)

    # construct new parents
    new_parents = []
    for i in range(len(pooling_list)):
        old_index = new_to_old[i]
        old_parent = parents[old_index]
        while not old_parent in old_to_new:
            old_parent = parents[old_parent]
        new_parents.append(old_to_new[old_parent])

    # add displacement
    if add_displacement:
        pooling_list.append([])
        for i in range(len(parents)):  # all joints use displacement in the unpooling
            pooling_list[-1].append(i)

    return pooling_list, new_parents


class SkeletonPool(nn.Module):
    def __init__(self, parents, channels_per_edge, device):
        super().__init__()

        self.device = device
        self.channels_per_edge = channels_per_edge

        self.pooling_list, self.new_parents = create_pooling_list(parents)

        self.description = "SkeletonPool(in_joints_num={}, out_joints_num={})".format(
            len(parents), len(self.pooling_list)
        )

        # rows: new joints, columns: old joints
        self.weight = torch.zeros(
            len(self.pooling_list) * channels_per_edge,
            len(parents) * channels_per_edge,
        ).to(self.device)

        # fill self.weights by recording the connections between old
        # and new joints. i are the indices of the new joint, j are
        # the indices of the old joints
        for i, merged in enumerate(self.pooling_list):
            for j in merged:
                for c in range(channels_per_edge):
                    self.weight[
                        i * channels_per_edge + c, j * channels_per_edge + c
                    ] = 1.0 / len(merged)

        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, input):
        return torch.matmul(self.weight, input)


class SkeletonUnpool(nn.Module):
    def __init__(self, pooling_list, channels_per_edge, device):
        super().__init__()

        self.device = device

        input_joints_num = len(pooling_list)
        out_joints = set()
        for merged in pooling_list:
            for j in merged:
                out_joints.add(j)
        output_joints_num = len(out_joints)

        self.description = "SkeletonUnpool(in_joints_num={}, out_joints_num={})".format(
            input_joints_num,
            output_joints_num,
        )

        self.weight = torch.zeros(
            output_joints_num * channels_per_edge, input_joints_num * channels_per_edge
        ).to(self.device)

        for i, merged in enumerate(pooling_list):
            for j in merged:
                for c in range(channels_per_edge):
                    self.weight[
                        j * channels_per_edge + c, i * channels_per_edge + c
                    ] = 1.0

        self.weight = nn.Parameter(self.weight, requires_grad=False)

    def forward(self, input):
        return torch.matmul(self.weight, input)


def find_collapse_joints(degrees, direct_neighbors):
    collapse_joints = []
    stack = [(0, -1)]  # start root
    visited = set()
    while len(stack) > 0:
        (curr, parent) = stack.pop()
        if curr == len(degrees):
            continue  # displacement is not a joint
        visited.add(curr)
        # if not root, parent has not been collapsed, and not a leaf node (end effector)
        if parent != -1 and parent not in collapse_joints and degrees[curr] > 1:
            # collapse
            collapse_joints.append(curr)
        # append children
        stack.extend(
            [
                (child, curr)
                for child in direct_neighbors[curr]
                if child != curr and child not in visited
            ]
        )
    return collapse_joints


def create_degree_list(parents):
    direct_neighbors, distance_mat = find_neighbor(parents, 1)
    degrees = np.zeros(len(parents))
    for i in range(len(parents)):
        for j in range(len(parents)):
            if distance_mat[i, j] == 1:
                degrees[i] += 1
    return degrees, direct_neighbors


def create_expanded_neighbor_list(neighbor_list, in_channels_per_joint):
    expanded_neighbor_list = []
    # create expanded_neighbor_list by appending channels of each neighbor joint
    for neighbor in neighbor_list:
        expanded = []
        for k in neighbor:
            for i in range(in_channels_per_joint):
                expanded.append(k * in_channels_per_joint + i)
        expanded_neighbor_list.append(expanded)
    return expanded_neighbor_list


def distance_joints(parents, i, j, init_dist=0):
    """
    Finds the distance between two joints if j is ancestor of i.
    Otherwise return 0.
    """
    if parents[i] == j:
        return init_dist + 1
    elif parents[i] == 0:
        return 0
    else:
        return distance_joints(parents, parents[i], j, init_dist + 1)


def calc_distance_mat(parents):
    """
    Parameters
    ----------
    parents : numpy.ndarray
        Parent indices of each joint.
    Returns
    -------
    distance_mat : numpy.ndarray
        Distance matrix len(parents) x len(parents) between any two joints
    """
    num_joints = len(parents)
    # distance_mat[i, j] = distance between joint i and joint j
    distance_mat = np.ones((num_joints, num_joints)) * np.inf
    for i in range(num_joints):
        distance_mat[i, i] = 0
    # calculate direct distances
    for i in range(num_joints):
        for j in range(num_joints):
            if i != j:
                d = distance_joints(parents, i, j)
                if d != 0:
                    distance_mat[i, j] = d
                    distance_mat[j, i] = d
    # calculate all other distances
    for k in range(num_joints):
        for i in range(num_joints):
            for j in range(num_joints):
                distance_mat[i][j] = min(
                    distance_mat[i][j], distance_mat[i][k] + distance_mat[k][j]
                )
    return distance_mat


def find_neighbor(parents, max_dist, add_displacement=False):
    distance_mat = calc_distance_mat(parents)
    neighbor_list = []  # for each joints contains all joints at max distance max_dist
    joint_num = len(parents)
    for i in range(joint_num):
        neighbor = []
        for j in range(joint_num):
            if distance_mat[i, j] <= max_dist:
                neighbor.append(j)
        neighbor_list.append(neighbor)

    # displacement is treated as another joint (appended)
    if add_displacement:
        displacement = joint_num
        displacement_neighbor = neighbor_list[0].copy()
        for i in displacement_neighbor:
            neighbor_list[i].append(displacement)
        displacement_neighbor.append(displacement)  # append itself
        neighbor_list.append(displacement_neighbor)

    return neighbor_list, distance_mat
