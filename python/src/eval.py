import os
import argparse
import torch
import train
from motion_data import TestMotionData
from train_data import Train_Data
from generator_architecture import Generator_Model
from ik_architecture import IK_Model
import eval_metrics

# Evaluation Modes
GENERATOR = 1
IK = 2


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_dataset = TestMotionData(train.param, train.scale, device)

    # Load BVH
    filename = os.path.basename(args.input_path)
    dir = args.input_path[: -len(filename)]
    rots, pos, parents, offsets, bvh = train.get_info_from_bvh(
        train.get_bvh_from_disk(dir, filename)
    )

    # Create Models
    train_data = Train_Data(device, train.param)
    generator_model = Generator_Model(device, train.param, parents, train_data).to(
        device
    )
    if args.eval_mode & IK != 0:
        ik_model = IK_Model(device, train.param, parents, train_data).to(device)

    # Load Models
    generator_model_path = os.path.join(args.model_path, "generator.pt")
    ik_model_path = os.path.join(args.model_path, "ik.pt")
    means, stds = train.load_model(
        generator_model, generator_model_path, train_data, device
    )
    if args.eval_mode & IK != 0:
        means, stds = train.load_model(ik_model, ik_model_path, train_data, device)

    # TestMotionData
    eval_dataset.set_means_stds(means, stds)
    eval_dataset.add_motion(
        offsets,
        pos[:, 0, :],  # only global position
        rots,
        parents,
        bvh,
        filename,
    )
    eval_dataset.normalize()

    results = train.evaluate_generator(generator_model, train_data, eval_dataset)
    if args.eval_mode & IK != 0:
        results_ik = train.evaluate_ik(ik_model, results, train_data, eval_dataset)
        results = results_ik

    # Save Result
    eval_path, eval_filename = train.result_to_bvh(
        results[0][0], means, stds, bvh, filename, save=True
    )

    # Evaluate Positional Error
    mpjpe, mpeepe = eval_metrics.eval_pos_error(
        train.get_bvh_from_disk(dir, filename),
        train.get_bvh_from_disk(eval_path, eval_filename),
        device,
    )

    print("Evaluate Loss: {}".format(mpjpe + mpeepe))
    print("Mean Per Joint Position Error: {}".format(mpjpe))
    print("Mean End Effector Position Error: {}".format(mpeepe))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Motion Upsampling Network")
    parser.add_argument(
        "model_path",
        type=str,
        help="path to pytorch model folder",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="path to the input .bvh file",
    )
    parser.add_argument(
        "eval_mode",
        type=str.lower,
        choices=["generator", "ik"],
        help="evaluation mode",
    )
    args = parser.parse_args()
    if args.eval_mode == "generator":
        args.eval_mode = GENERATOR
    elif args.eval_mode == "ik":
        args.eval_mode = IK
    main(args)
