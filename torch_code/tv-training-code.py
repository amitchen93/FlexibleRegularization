# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import argparse
import os
import numpy as np
import torch
from PIL import Image

import torchvision
import attr
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
from torch_code.pytorch_adaptive_optim import SGD

print("i forgot to delete this print in tv training code")
class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

@attr.s
class TVFinetune:
    # def __init__(self, args):
    args = attr.ib()
    task = attr.ib(None)


    def __attrs_post_init__(self):
        self.logger = None
        self.num_classes = 2
        self.root = self.args.root  # "/Users/hyamsga/Projects/others/un/Data/PennFudanPed"
        self.device = torch.device('cuda') if torch.cuda.is_available() and self.args.gpu else torch.device('cpu')
        # self.device = torch.device('cpu')  # torch.device('cuda') if torch.cuda.is_available() and self.args.gpu else torch.device('cpu')
        # weight_decay = 0.0005
        if self.args.trains:
            from trains import Task
            self.task = Task.init(project_name='Flexible Regularization',
                             task_name='Finetune Maskrcnn')  # 'Torch Models')
        else:
            self.task = None
        # task = attr.ib(None)
        self.logger = self.task.current_task().get_logger() if self.task else None  # self.args.trains else None
        self.train_data_loader, self.test_data_loader = self.get_dataloaders()

    # def execute(self):
        # train on the GPU or on the CPU, if a GPU is not available
        # our dataset has two classes only - background and person
        # use our dataset and defined transformations
        # dataset = PennFudanDataset(root, get_transform(train=True))
        # dataset_test = PennFudanDataset(root, get_transform(train=False))
        #
        # # split the dataset in train and test set
        # indices = torch.randperm(len(dataset)).tolist()
        # dataset = torch.utils.data.Subset(dataset, indices[:-50])
        # dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        #
        # # define training and validation data loaders
        # data_loader = torch.utils.data.DataLoader(
        #     dataset, batch_size=2, shuffle=True, num_workers=4,
        #     collate_fn=utils.collate_fn)
        #
        # data_loader_test = torch.utils.data.DataLoader(
        #     dataset_test, batch_size=1, shuffle=False, num_workers=4,
        #     collate_fn=utils.collate_fn)

        # get the model using our helper function
        # model = get_model_instance_segmentation(num_classes)
        #
        # # move model to the right device
        # model.to(device)
        #
        # # construct an optimizer
        # params = [p for p in model.parameters() if p.requires_grad]
        # if not self.args.adaptive_var_reg and not self.args.adaptive_avg_reg:
        #     optimizer = torch.optim.SGD(params, lr=0.005,
        #                                 momentum=0.9, weight_decay=0.0005)
        # else:
        #     optimizer = SGD(params, lr=0.005,
        #                     momentum=0.9, weight_decay=0.0005,
        #                     nesterov=self.args.neterov, adaptive_var_weight_decay=self.args.adaptive_var_reg,
        #                     iter_length=self.args.iter_length, device=device,
        #                     inverse_var=self.args.inverse_var, adaptive_avg_reg=self.args.adaptive_avg_reg,
        #                     logger=self.logger, static_var_calculation=self.args.static_var_calculation,
        #                     uniform_prior_strength=0.5
        #                     )
        # and a learning rate scheduler

        # optimizers = {
        #     "adaptive wd optimizer": SGD(params, lr=0.005,
        #                             momentum=0.9, weight_decay=weight_decay),
        #     "regular optimizer": torch.optim.SGD(params, lr=0.005,
        #                                          momentum=0.9, weight_decay=weight_decay)
        # }
        # for optimizer_name, optimizer in optimizers.items():
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
            #                                                step_size=3,
            #                                                gamma=0.1)
            # # let's train it for 10 epochs
            # num_epochs = self.args.epochs # 10
            #
            # for epoch in range(num_epochs):
            #     # train for one epoch, printing every 10 iterations
            #     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            #     # update the learning rate
            #     lr_scheduler.step()
            #     # evaluate on the test dataset
            #     evaluate(model, data_loader_test, device=device)

            # print("That's it!")

    def get_dataloaders(self):
        dataset = PennFudanDataset(self.root, get_transform(train=True))
        dataset_test = PennFudanDataset(self.root, get_transform(train=False))

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        # define training and validation data loaders
        train_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        test_data_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)
        return train_data_loader, test_data_loader

    def train_and_eval(self):
        model = get_model_instance_segmentation(self.num_classes)

        # move model to the right device
        model.to(self.device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if not self.args.adaptive_var_reg and not self.args.adaptive_avg_reg:
            optimizer = torch.optim.SGD(params, lr=0.005,
                                        momentum=0.9, weight_decay=self.args.reg_strength)
        else:
            optimizer = SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=self.args.reg_strength,
                            nesterov=self.args.neterov, adaptive_var_weight_decay=self.args.adaptive_var_reg,
                            iter_length=self.args.iter_length, device=self.device,
                            inverse_var=self.args.inverse_var, adaptive_avg_reg=self.args.adaptive_avg_reg,
                            logger=self.logger, static_var_calculation=self.args.static_var_calculation,
                            uniform_prior_strength=0.5
                            )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)
        # let's train it for 10 epochs
        num_epochs = self.args.epochs  # 10

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, self.train_data_loader, self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, self.test_data_loader, device=self.device)


def clean_gpu():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # clean_gpu()
    args = parse_args()
    print(args)
    tv_fintune = TVFinetune(args)
    tv_fintune.train_and_eval()


def parse_args():
    parser = argparse.ArgumentParser(description='adaptive regularization')
    parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--root', default="/Users/hyamsga/Projects/others/un/Data/PennFudanPed")
    # parser.add_argument("--print_every", type=int, default=10)
    # parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--iter_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--model", default='mlp') #, choices=['mlp', 'cnn', 'alexnet',])
    # parser.add_argument("--num_trains", default=49000, type=int)
    # parser.add_argument("--num_of_repeats", default=1, type=int)
    # parser.add_argument("--dropconnect", default=1, type=float)
    parser.add_argument("--adaptive_var_reg", default=1, type=int)
    parser.add_argument("--reg_strength", default=0.0005, type=float)
    # parser.add_argument("--adaptive_dropconnect", default=0, type=int)
    # parser.add_argument("--divide_var_by_mean_var", default=1, type=int)
    # parser.add_argument("--test", default=0, type=int)
    # parser.add_argument("--variance_calculation_method", default="welford", choices=["naive", "welford", "GMA"])
    # parser.add_argument("--static_variance_update", default=1, type=int)
    # parser.add_argument("--var_normalizer", default=1, type=float)  # todo: make sure this is the right value to put
    # parser.add_argument("--batchnorm", default=0, type=int, help="Available only for MLP.")
    # parser.add_argument("--optimizer", default='sgd', choices=['sgd', 'sgd_momentum', 'adam', 'rmsprop', None])
    # parser.add_argument("--baseline_as_well", default=1, type=int)
    # parser.add_argument("--eval_distribution_sample", default=0, type=float)
    parser.add_argument("--inverse_var", default=1, type=int)
    parser.add_argument("--adaptive_avg_reg", default=0, type=int)
    # parser.add_argument("--mean_mean", default=0, type=int)
    parser.add_argument("--trains", default=1, type=int)
    # parser.add_argument("--hidden_layers", default=5, type=int)
    # parser.add_argument("--lnn", default=0, type=int)
    # parser.add_argument("--reg_layers", default='1,2,3')
    # parser.add_argument("--momentum", type=int, default=0)
    parser.add_argument("--neterov", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--pretrained", type=int, default=1)
    # parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--static_var_calculation", type=int, default=1)
    # parser.add_argument("--scheduler", type=float, default=0)
    # parser.add_argument("--dataset", default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    # parser.add_argument("--output_dir", default=Path('/cs/labs/gavish/gal.hyams/data/out/dr'))    #  /tmp))  # /cs/labs/gavish/gal.hyams/data/out/dr
    # parser.add_argument("--uniform_prior_strength", type=float, default=0)
    # parser.add_argument("--knn_class_ratio", type=float, default=0.5, help="seen classes / all classes")
    # parser.add_argument("--weight_decay_decay", type=float, default=1.0)
    # parser.add_argument("--random_train_val", type=float, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
