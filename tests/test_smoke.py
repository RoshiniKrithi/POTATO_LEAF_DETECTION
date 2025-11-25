import torch

from potato_leaf_detection.models.factory import create_model


def test_model_forward():
    model, sz = create_model("efficientnet_b0", num_classes=3, pretrained=False)
    x = torch.randn(2, 3, sz, sz)
    y = model(x)
    assert y.shape == (2, 3)


