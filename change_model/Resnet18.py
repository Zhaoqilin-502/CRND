import torch, timm
from thop import clever_format, profile
def Resnet18():
    model = timm.create_model('resnet18', pretrained=False, features_only=True)
    return model
