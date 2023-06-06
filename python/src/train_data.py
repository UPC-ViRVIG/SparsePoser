import torch


class Train_Data:
    def __init__(self, device, param):
        super().__init__()
        self.device = device
        self.param = param
        self.losses = []

    def set_offsets(self, offsets, denorm_offsets):
        self.offsets = offsets
        self.denorm_offsets = denorm_offsets
        for loss in self.losses:
            loss.set_offsets(denorm_offsets[0])

    def set_motions(self, dqs, displacement):
        # concatenate the displacement to the dqs
        self.motion = torch.cat([dqs, displacement], dim=2)
        # swap second and third dimensions for convolutions (last row should be time)
        self.motion = self.motion.permute(0, 2, 1)
        # self.motion is tensor of shape (batch_size, n_joints*8 + 3, frames)
        # if time dimension is not multiple of 8... make it multiple of 8
        # so that convolutions always return an even number (3 time downsamplings)
        if self.motion.shape[2] % 8 != 0:
            self.motion = self.motion[
                :, :, : self.motion.shape[2] - self.motion.shape[2] % 8
            ]
        # input for sparse encoder
        # sparse_dqs (batch_size, n_sparse_joints, 8, frames)
        sparse_dqs = (
            self.motion[:, :-3, :]
            .reshape(self.motion.shape[0], -1, 8, self.motion.shape[-1])
            .clone()
        )
        sparse_dqs = sparse_dqs[:, self.param["sparse_joints"], ...]
        sparse_dqs = sparse_dqs.flatten(start_dim=1, end_dim=2)
        # sparse_displacement (batch_size, 3, frames)
        sparse_displacement = self.motion[:, -3:, :].clone()
        # self.sparse_motion (batch_size, n_sparse_joints * 8 + 3, frames)
        self.sparse_motion = torch.cat([sparse_dqs, sparse_displacement], dim=1)
        # remove displacement from self.motion (we use it only as input)
        # self.motion is tensor of shape (batch_size, n_joints*8, frames)
        self.motion = self.motion[:, :-3, :]

    def set_sparse_motion(self, sparse_motion):
        self.sparse_motion = sparse_motion

    def set_means(self, mean_dqs):
        self.mean_dqs = mean_dqs
        for loss in self.losses:
            loss.set_mean(mean_dqs)

    def set_stds(self, std_dqs):
        self.std_dqs = std_dqs
        for loss in self.losses:
            loss.set_std(std_dqs)
