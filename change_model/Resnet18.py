import torch, timm
from thop import clever_format, profile
def Resnet18():
    model = timm.create_model('resnet18', pretrained=False, features_only=True)
    return model
#
# print(timm.list_models())
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dummy_input = torch.randn(1, 3, 640, 640).to(device)
# model = Resnet18()
# model.to(device)
# model.eval()
# print(model.feature_info.channels())
# for feature in model(dummy_input):
#     print(feature.size())
#
# flops, params = profile(model.to(device), (dummy_input,), verbose=False)
# flops, params = clever_format([flops * 2, params], "%.3f")
# print('Total FLOPS: %s' % (flops))
# print('Total params: %s' % (params))