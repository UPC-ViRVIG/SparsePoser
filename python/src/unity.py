import os
import struct
import torch
import socket
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model
import train
import argparse
import numpy as np
import dual_quat as dquat
from motion_data import RunMotionData

HOST = "127.0.0.1"
PORT = 2222
WINDOW = 64  # same number as C# code
SENT_POSE_INDEX = -1

# Run Modes
GENERATOR = 1
IK = 2


def read_float_array(bytes, floats):
    for i in range(len(floats)):
        floats[i] = struct.unpack("<f", bytes[i * 4 : i * 4 + 4])[0]  # little endian


def write_float_array(floats):
    return struct.pack("<{}f".format(len(floats)), *floats)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BVH
    filename = os.path.basename(args.reference_bvh)
    dir = args.reference_bvh[: -len(filename)]
    template_rots, template_pos, parents, offsets, _ = train.get_info_from_bvh(
        train.get_bvh_from_disk(dir, filename)
    )

    # Create Models
    train_data = Train_Data(device, train.param)
    generator_model = Generator_Model(device, train.param, parents, train_data).to(
        device
    )
    if args.run_mode & IK != 0:
        ik_model = IK_Model(device, train.param, parents, train_data).to(device)

    # Load Models
    generator_model_path = os.path.join(args.model_path, "generator.pt")
    ik_model_path = os.path.join(args.model_path, "ik.pt")
    means, stds = train.load_model(
        generator_model, generator_model_path, train_data, device
    )
    if args.run_mode & IK != 0:
        means, stds = train.load_model(ik_model, ik_model_path, train_data, device)

    # Auxiliar Dataset Structure
    run_dataset = RunMotionData(train.param, device)
    run_dataset.set_means_stds(means, stds)
    run_dataset.set_offsets(offsets)
    run_dataset.normalize_offsets()

    # create a socket for IPv4 and TCP
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))  # associate socket with interface en port number
            s.listen()
            print("socket listening...")
            (
                conn,
                addr,
            ) = (
                s.accept()
            )  # blocks and waits for a connection, conn is a new socket, addr is (host, port)
            with conn:
                print("Connected by {}".format(addr))
                while True:
                    data = conn.recv(WINDOW * len(train.param["sparse_joints"]) * 7 * 4)
                    if (
                        not data
                    ):  # if empty byte object b'' is returned -> client closed the connection
                        break
                    # convert data bytes to actual positions and rotations
                    floats = np.empty((WINDOW * len(train.param["sparse_joints"]) * 7))
                    read_float_array(data, floats)
                    pos = np.zeros((WINDOW, template_pos.shape[1], 3))
                    rot = np.zeros((WINDOW, template_rots.shape[1], 4))
                    rot[:, :, 0] = 1.0  # valid quaternions
                    for f in range(WINDOW):
                        for j_index, j in enumerate(train.param["sparse_joints"]):
                            offset = f * len(train.param["sparse_joints"]) * (
                                3 + 4
                            ) + j_index * (3 + 4)
                            pos[f, j, 0] = floats[offset + 0]
                            pos[f, j, 1] = floats[offset + 1]
                            pos[f, j, 2] = floats[offset + 2]
                            rot[f, j, 0] = floats[offset + 3]
                            rot[f, j, 1] = floats[offset + 4]
                            rot[f, j, 2] = floats[offset + 5]
                            rot[f, j, 3] = floats[offset + 6]
                    # Fill RunMotionData
                    run_dataset.set_motion(pos, rot)
                    run_dataset.normalize_motion()
                    # Run
                    train.run_set_data(train_data, run_dataset)
                    res = train.run_generator(generator_model)
                    if args.run_mode & IK != 0:
                        res = train.run_ik(ik_model, res, SENT_POSE_INDEX)
                    res = res[:, :, SENT_POSE_INDEX].unsqueeze(-1)
                    # Convert to rotations
                    res = res.permute(0, 2, 1)
                    res = res.flatten(0, 1)
                    res = res.cpu().detach().numpy()
                    dqs = res
                    # denormalize
                    dqs = dqs * stds["dqs"].cpu().numpy() + means["dqs"].cpu().numpy()
                    # get rotations and translations from dual quatenions
                    dqs = dqs.reshape(dqs.shape[0], -1, 8)
                    rots, _ = dquat.skeleton_from_dual_quat(dqs, parents)
                    # rots (frames, joints, [w,x,y,z])
                    # convert to array of bytes
                    send_data = write_float_array(rots[0, :, :].flatten())
                    conn.sendall(send_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCP Connection with Unity")
    parser.add_argument(
        "model_path",
        type=str,
        help="path to pytorch model folder",
    )
    parser.add_argument(
        "reference_bvh",
        type=str,
        help="path to the reference .bvh file",
    )
    parser.add_argument(
        "run_mode",
        type=str.lower,
        choices=["generator", "ik"],
        help="run mode",
    )
    args = parser.parse_args()
    if args.run_mode == "generator":
        args.run_mode = GENERATOR
    elif args.run_mode == "ik":
        args.run_mode = IK
    main(args)
