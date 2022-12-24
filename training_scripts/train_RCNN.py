import argparse
from pathlib import Path
import os
import wandb

from utils.dataset.dataset import build_dataloaders
from utils.vision.engine import train_one_epoch, evaluate

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model_type = "RCNN"


def train(args):

    if not os.path.isdir("results/" + model_type + "/"):
        os.makedirs("results/" + model_type + "/")

    train_loader, test_loader, num_classes = build_dataloaders(args.wanted_classes)

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device %s" % device)

    # move model to the right device
    model.to(device)

    if args.init_save_path is not None:
        model.load_state_dict(torch.load(args.init_save_path))

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.nb_epochs):
        to_log = {}

        # train for one epoch, printing every 10 iterations
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=1000
        )

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        test_metric = evaluate(model, test_loader, device=device)
        test_coco_eval = test_metric.coco_eval["bbox"]

        # for the wandb logs, we take the metrics with [ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        precision = test_coco_eval.stats[0]
        recall = test_coco_eval.stats[8]
        f1_score = 2 * precision * recall / (precision + recall)

        to_log["main/train_loss"] = train_metrics.loss.value
        to_log["main/precision"] = precision
        to_log["main/recall"] = recall
        to_log["main/f1_score"] = f1_score
        wandb.log(to_log)
        torch.save(
            model.state_dict(),
            "results/" + model_type + "/" + args.wanted_classes + ".pt",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb or not.",
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
        default=10,
        help="Number of epochs.",
    )

    parser.add_argument(
        "--init_save_path",
        type=str,
        default=None,
        help="Path to the pretrained agent. It is use if it is not None",
    )

    args = parser.parse_args()

    train(args)
