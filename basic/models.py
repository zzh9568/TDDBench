"""This file contains the definition of models"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from wide_resnet import WideResNet
from resnet import ResNet, BasicBlock, Bottleneck
from mobilenet import MobileNetV2
from vgg import VGG

INPUT_OUTPUT_SHAPE = {
    "cifar10": [3, 10],
    "cifar10-demo": [3, 10],
    "cifar100": [3, 100],
    "celeba": [3, 2],
    "lfw": [3, 5749],

    "pathmnist": [3, 9],
    "chestmnist": [3, 14],
    "octmnist": [3, 4],
    "breastmnist": [3, 2],
    "dermamnist": [3, 7],
    "retinamnist": [3, 5],
    "bloodmnist": [3, 8],
    "organamnist": [3, 11],

    "purchase100": [600, 100],
    "texas100": [6169, 100],
    "breast_cancer": [30, 2],
    "student": [32, 3],
    "adult": [7, 2],

    "tweet_eval_hate": 2, 
    "rotten_tomatoes": 2,
    "cola": 2,
    "ecthr_articles": 13,
    "ag_news": 4,
}


class NN(nn.Module):

    def __init__(self, in_shape, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_shape, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = inputs.flatten(1)
        inputs = torch.tanh(self.fc1(inputs))
        outputs = self.fc2(inputs)
        return outputs

class MLP(nn.Module):

    def __init__(self, in_shape, hiddens=[512, 256, 128, 64], num_classes=10):
        super().__init__()
        self.hiddens = [in_shape] + hiddens + [num_classes]
        self.layers = torch.nn.ModuleList()
        for i in range(0, len(self.hiddens)-1):
            self.layers.append(nn.Linear(self.hiddens[i], self.hiddens[i+1]))

    def forward(self, inputs):
        """Forward pass of the model."""
        h = inputs.flatten(1)

        for i in range(0, len(self.layers)-1):
            h = torch.tanh(self.layers[i](h))

        outputs = self.layers[-1](h)
        return outputs
    

class LeNet(nn.Module):
    """Simple CNN for CIFAR10 dataset."""

    def __init__(self, num_classes=10, num_channels=16):
        super().__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, num_channels, 5)
        self.fc1 = nn.Linear(num_channels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = self.pool(F.relu(self.conv1(inputs)))
        inputs = self.pool(F.relu(self.conv2(inputs)))
        # flatten all dimensions except batch
        inputs = inputs.reshape(-1, self.num_channels * 5 * 5)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        outputs = self.fc3(inputs)
        return outputs


class AlexNet(nn.Module):
    """AlexNet model for CIFAR10 dataset."""

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, inputs):
        """Forward pass of the model."""
        inputs = self.features(inputs)
        inputs = inputs.reshape(inputs.size(0), 256 * 2 * 2)
        outputs = self.classifier(inputs)
        return outputs


def get_model(model_type: str, dataset_name: str, num_classes: int=None):
    """Instantiate the model based on the model_type

    Args:
        model_type (str): Name of the model
        dataset_name (str): Name of the dataset
    Returns:
        torch.nn.Module: A model
    """
    if num_classes is None:
        num_classes = INPUT_OUTPUT_SHAPE[dataset_name][1]
    in_shape = INPUT_OUTPUT_SHAPE[dataset_name][0]
    if model_type[:3] == "cnn":
        return LeNet(num_classes=num_classes, num_channels=int(model_type[3:]))
    if model_type == "lenet":
        return LeNet(num_classes=num_classes)
    elif model_type == "alexnet":
        return AlexNet(num_classes=num_classes)
    elif model_type == "wrn28-1":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=1)
    elif model_type == "wrn28-2":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=2)
    elif model_type == "wrn28-10":
        return WideResNet(nin=in_shape, nclass=num_classes, depth=28, width=10)
    elif model_type == "nn":
        # for tabular datasets
        return NN(in_shape=in_shape, num_classes=num_classes)
    elif model_type == "mlp":
        # for tabular datasets
        return MLP(in_shape=in_shape, num_classes=num_classes)
    elif model_type == "mlp1":
        return MLP(in_shape=in_shape, hiddens=[64], num_classes=num_classes)
    elif model_type == "mlp2":
        return MLP(in_shape=in_shape, hiddens=[128, 64], num_classes=num_classes)
    elif model_type == "mlp3":
        return MLP(in_shape=in_shape, hiddens=[256, 128, 64], num_classes=num_classes)
    elif model_type == "mlp5":
        return MLP(in_shape=in_shape, hiddens=[1024, 512, 256, 128, 64], num_classes=num_classes)
    elif model_type == "mlp16":
        return MLP(in_shape=in_shape, hiddens=[16], num_classes=num_classes)
    elif model_type == "mlp32":
        return MLP(in_shape=in_shape, hiddens=[32], num_classes=num_classes)
    elif model_type == "mlp64":
        return MLP(in_shape=in_shape, hiddens=[64], num_classes=num_classes)
    elif model_type == "mlp128":
        return MLP(in_shape=in_shape, hiddens=[128], num_classes=num_classes)
    elif model_type == "mlp256":
        return MLP(in_shape=in_shape, hiddens=[256], num_classes=num_classes)
    elif model_type in ["vgg16", "vgg11", "mobilenet-v2", "densenet121", "inception-v3", "resnet10", "resnet18", "resnet34", "resnet50"]:
        if model_type == "vgg16":
            model = VGG(vgg_name="VGG16", num_classes=num_classes)
        elif model_type == "vgg11":
            model = VGG(vgg_name="VGG11", num_classes=num_classes)
        elif model_type == "inception-v3":
            model = torchvision.models.inception_v3(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type == "mobilenet-v2":
            model = MobileNetV2(class_num=num_classes)
        elif model_type == "densenet121":
            model = torchvision.models.densenet121(pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model_type == "resnet10":
            model = ResNet(BasicBlock, [1,1,1,1], num_classes=num_classes)
        elif model_type == "resnet18":
            model = ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)
        elif model_type == "resnet34":
            model = ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)
        elif model_type == "resnet50":
            model = ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)
        elif model_type == "resnet101":
            model = ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)
        elif model_type == "resnet152":
            model = ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)
        else:
            raise NotImplementedError(f"{model_type} is not implemented")
        return model
    else:
        raise NotImplementedError(f"{model_type} is not implemented")
    



