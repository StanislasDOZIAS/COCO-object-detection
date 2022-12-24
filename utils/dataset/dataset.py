from PIL import Image
import torch

import fiftyone.utils.coco as fouc
import fiftyone.zoo as foz


import utils.vision.utils as utils
import utils.vision.transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def build_dataloaders(wanted_classes: str, batch_size: int = 1):
    fo_train_set = create_fiftyone_dataset(
        wanted_classes, "train", drop_existing_dataset=False
    )
    fo_test_set = create_fiftyone_dataset(
        wanted_classes, "validation", drop_existing_dataset=False
    )

    fo_train_set.compute_metadata()
    fo_test_set.compute_metadata()

    train_set = FiftyOneTorchDataset(fo_train_set, get_transform(train=True))
    test_set = FiftyOneTorchDataset(fo_test_set, get_transform(train=False))
    num_classes = len(train_set.classes)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=utils.collate_fn,
    )

    return train_loader, test_loader, num_classes



class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    Comes from https://github.com/voxel51/fiftyone-examples/blob/master/examples/pytorch_detection_training.ipynb

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """

    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct("%s.detections.label" % gt_field)

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        area = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            category_id = self.labels_map_rev[det.label]
            coco_obj = fouc.COCOObject.from_label(
                det,
                metadata,
                category_id=category_id,
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            area.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes


def create_fiftyone_dataset(wanted_classes, split=None, drop_existing_dataset=False):
    """
    Create a fiftyone dataset with the images
    """
    if split is None:
        dataset_name = wanted_classes
    else:
        dataset_name = split + "_" + wanted_classes

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        classes=class_filter[wanted_classes],
        split=split,
        dataset_name=dataset_name,
        only_matching=True,
        drop_existing_dataset=drop_existing_dataset,
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
