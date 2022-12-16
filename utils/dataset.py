import fiftyone
import json


def create_fiftyone_dataset(wanted_classes):
    dataset = fiftyone.zoo.load_zoo_dataset(
        "coco-2017", classes=class_filter[wanted_classes]
    )
    return dataset


class_filter = {
    "vehicles": [
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ],
    "signals": [
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
    ],
    "animals": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "objects": [
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
    ],
    "sport_objects": [
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
    ],
    "ustensils": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "food": [
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    ],
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "tech": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "house": [
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ],
    "others": [
        "__background__",
        "person",
    ],
}
