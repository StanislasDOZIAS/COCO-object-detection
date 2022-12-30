import argparse
from pathlib import Path
import os
import wandb

from utils.dataset.dataset import build_dataloaders
from utils.vision.engine import train_one_epoch, evaluate

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model_type = "faster_RCNN"


def train(args):

    if args.name is None:
        print("You must pass a name for the run")
        raise ValueError

    res_model_path = Path("results/") / model_type
    if not res_model_path.is_dir():
        os.makedirs(res_model_path)

    res_wanted_classes_path = res_model_path / args.wanted_classes
    if not res_wanted_classes_path.is_dir():
        os.makedirs(res_wanted_classes_path)

    res_experience_path = res_wanted_classes_path / args.name
    if not res_experience_path.is_dir():
        os.makedirs(res_experience_path)
    else:
        print()
        print("#" * len("CAREFULL, YOU WILL ERASE OLD WEIGHTS"))
        print("#" * len("CAREFULL, YOU WILL ERASE OLD WEIGHTS"))
        print("CAREFULL, YOU WILL ERASE OLD WEIGHTS")
        print("#" * len("CAREFULL, YOU WILL ERASE OLD WEIGHTS"))
        print("#" * len("CAREFULL, YOU WILL ERASE OLD WEIGHTS"))
        print()

    name = model_type + "/" + args.wanted_classes + "/" + args.name

    if args.run_id is not None:
        wandb.init(
            project="Object Detection",
            name=name,
            id=args.run_id,
            resume="must",
        )
    else:
        wandb.init(project="Object Detection", name=name)

    train_loader, test_loader, num_classes = build_dataloaders(args.wanted_classes)

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # get the trained weights
    if args.init_save_path is not None:
        print("loading weights from", args.init_save_path)
        model.load_state_dict(torch.load(args.init_save_path))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device %s" % device)

    # move model to the right device
    model.to(device)

    if args.run_id is None:
        to_log = {}
        # evaluate on the test dataset for a first evaluation
        test_metric = evaluate(model, test_loader, device=device)
        to_log["main/mAP"] = test_metric.coco_eval["bbox"].stats[0]
        wandb.log(to_log)

    for epoch in range(args.nb_epochs):
        to_log = {}

        # train for one epoch, printing every 1000 iterations
        train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            print_freq=1000,
            # wanted_losses=["loss_classifier", "loss_box_reg"],
            wanted_losses=None,
        )

        # evaluate on the test dataset
        test_metric = evaluate(model, test_loader, device=device)

        # for the wandb logs, we take the mean Average Precision : Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        to_log["main/mAP"] = test_metric.coco_eval["bbox"].stats[0]

        wandb.log(to_log)
        torch.save(
            model.state_dict(),
            "results/" + name + "/model.pt",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb or not.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name for wandb. Only used if not None",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run id to append logs to a previous wandb run. Only used if not None",
    )

    parser.add_argument(
        "--wanted_classes",
        type=str,
        default="signals",
        help="classes to train on.",
    )

    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=5,
        help="Number of epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )

    parser.add_argument(
        "--init_save_path",
        type=str,
        default=None,
        help="Path to the pretrained agent. Only used if not None",
    )

    args = parser.parse_args()

    train(args)
