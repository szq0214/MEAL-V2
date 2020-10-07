dependencies = [
    'timm',
    'torch',
]

import torch, timm

__all__ = ['mealv1_resnest50', 'mealv2_resnest50', 'mealv2_resnest50_cutmix', 'mealv2_resnest50_380x380', 'mealv2_mobilenetv3_small_075', 'mealv2_mobilenetv3_small_100', 'mealv2_mobilenet_v3_large_100', 'mealv2_efficientnet_b0']

model_urls = {
	  'mealv1_resnest50': 'https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV1_ResNet50_224.pth',
    'mealv2_resnest50': 'https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV2_ResNet50_224.pth',
    'mealv2_resnest50_cutmix': 'https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV2_ResNet50_224_cutmix.pth',
    'mealv2_resnest50_380x380': 'https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV2_ResNet50_380.pth',
    'mealv2_mobilenetv3_small_075': 'https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV2_MobileNet_V3_Small_0.75_224.pth',
    'mealv2_mobilenetv3_small_100': 'https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV2_MobileNet_V3_Small_1.0_224.pth',
    'mealv2_mobilenet_v3_large_100': 'https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV2_MobileNet_V3_Large_1.0_224.pth',
    'mealv2_efficientnet_b0': 'https://github.com/szq0214/MEAL-V2/releases/download/v1.0.0/MEALV2_EfficientNet_B0_224.pth',
}


mapping = {'mealv1_resnest50':'resnet50', 
           'mealv2_resnest50':'resnet50', 
           'mealv2_resnest50_cutmix':'resnet50', 
           'mealv2_resnest50_380x380':'resnet50', 
           'mealv2_mobilenetv3_small_075':'tf_mobilenetv3_small_075', 
           'mealv2_mobilenetv3_small_100':'tf_mobilenetv3_small_100', 
           'mealv2_mobilenet_v3_large_100':'tf_mobilenetv3_large_100', 
           'mealv2_efficientnet_b0':'tf_efficientnet_b0'
}

def meal_v2(model_name, pretrained=True, progress=True):
    """ MEAL V2 models from
    `"MEAL V2: Boosting Vanilla ResNet-50 to 80%+ Top-1 Accuracy on ImageNet without Tricks" <https://arxiv.org/pdf/2009.08453.pdf>`_

    Args:
        model_name: Name of the model to load
        pretrained (bool): If True, returns a model trained with MEAL V2 on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    model = timm.create_model(mapping[model_name.lower()], pretrained=False)
    if pretrained:
       state_dict = torch.hub.load_state_dict_from_url(model_urls[model_name.lower()], progress=progress)
       model = torch.nn.DataParallel(model).cuda()
       model.load_state_dict(state_dict)
    return model