import torch
import torch.nn as nn
from torchvision import models as torch_models

from octopod.vision.helpers import _dense_block, _Identity


class ResNetStartingBlock(nn.Module):
    """
    ResNet, dense layers, and a final linear layer. This should serve as the starting block for the
    CNN, specifically for the first level.

    Architecture is as follows:
    * ResNet50
    * Three dense blocks
    * Linear layer

    Parameters
    ----------
    task_size: int
        The number of tasks to output
    load_pretrained_resnet: bool
        Flag for whether or not to load in pretrained weights for ResNet50. This is useful for the
        first round of training before there are fine-tuned weights (default True)

    """
    def __init__(self, task_size, load_pretrained_resnet=False):
        super(ResNetStartingBlock, self).__init__()

        self.resnet = torch_models.resnet50(pretrained=load_pretrained_resnet)
        self.resnet.fc = _Identity()

        self.dense_layers = nn.Sequential(
            _dense_block(2048*2, 1024, 2e-3),
            _dense_block(1024, 512, 2e-3),
            _dense_block(512, 256, 2e-3),
        )

        self.classifier = nn.Linear(256, task_size)
        # for key, task_size in new_task_dict.items():
        #   new_layers[key] = nn.Linear(256, task_size)

    def forward(self, x):
        """
        Forward pass for the model through all layers.

        Parameters
        ----------
        x: torch.tensor, 2-d
            Input should be of shape `batch_size * start`

        Returns
        ----------
        output: torch.tensor, 2-d
            Output will be of shape `batch_size * task_size`

        """
        dense_layer_output = self.get_embeddings(x)

        return self.classifier(dense_layer_output)

    def get_embeddings(self, x):
        """
        Forward pass for the model through all layers EXCEPT the final linear layer. This method
        returns the penultimate embedding layer rather than the actual prediction.

        Parameters
        ----------
        x: torch.tensor, 2-d
            Input should be of shape `batch_size * start`

        Returns
        ----------
        output: torch.tensor, 2-d
            Output will be of shape `batch_size * 256`

        """
        full_img = self.resnet(x['full_img']).squeeze()
        crop_img = self.resnet(x['crop_img']).squeeze()

        if x[next(iter(x))].shape[0] == 1:
            # if batch size is 1, or a single image, during inference
            full_crop_combined = torch.cat((full_img, crop_img), 0).unsqueeze(0)
        else:
            full_crop_combined = torch.cat((full_img, crop_img), 1)

        dense_layer_output = self.dense_layers(full_crop_combined)

        return dense_layer_output


class DenseBlock(nn.Module):
    """
    Dense layers and a final linear layer that accepts an embedding layer from another model as
    input. This should serve as the additional blocks for the CNN, specifically for the level later
    than the first.

    Architecture is as follows:
    * Two dense blocks
    * Linear layer

    Parameters
    ----------
    task_size: int
        The number of tasks to output
    start: int
        Dense layer starting dimension
    end: int
        Dense layer ending dimension for the final embedding, before the linear layer output of size
        `task_size`

    """
    def __init__(self,
                 task_size,
                 start=None,
                 end=None):
        super(DenseBlock, self).__init__()

        middle = int((start + end) / 2)

        self.dense_layers = nn.Sequential(
            _dense_block(start, middle, 2e-3),
            _dense_block(middle, end, 2e-3),
        )

        self.classifier = nn.Linear(end, task_size)

    def forward(self, x):
        """
        Forward pass for the model through all layers.

        Parameters
        ----------
        x: torch.tensor, 2-d
            Input should be of shape `batch_size * start`

        Returns
        ----------
        output: torch.tensor, 2-d
            Output will be of shape `batch_size * task_size`

        """
        dense_layer_output = self.get_embeddings(x)

        return self.classifier(dense_layer_output)

    def get_embeddings(self, x):
        """
        Forward pass for the model through all layers EXCEPT the final linear layer. This method
        returns the penultimate embedding layer rather than the actual prediction.

        Parameters
        ----------
        x: torch.tensor, 2-d
            Input should be of shape `batch_size * start`

        Returns
        ----------
        output: torch.tensor, 2-d
            Output will be of shape `batch_size * end`

        """
        dense_layer_output = self.dense_layers(x)

        return dense_layer_output
