# 224 x 224 ResNet-50 Tested on 8 TITAN Xp GPUs
python train.py --save MEAL_V2_resnet50_224 --batch-size 512 -j 48 --model resnet50 --teacher-model gluon_senet154,gluon_resnet152_v1s --imagenet [imagenet-folder with train and val folders] 

# # 224 x 224 MobileNet V3-Small 0.75
# python train.py --save MEAL_V2_mobilenetv3_small_075 --batch-size 512 -j 48 --model tf_mobilenetv3_small_075 --teacher-model gluon_senet154,gluon_resnet152_v1s --imagenet [imagenet-folder with train and val folders] 

# # 224 x 224 MobileNet V3-Small 1.0
# python train.py --save MEAL_V2_mobilenetv3_small_100 --batch-size 512 -j 48 --model tf_mobilenetv3_small_100 --teacher-model gluon_senet154,gluon_resnet152_v1s --imagenet [imagenet-folder with train and val folders]

# # 224 x 224 MobileNet V3-Large 1.0
# python train.py --save MEAL_V2_mobilenetv3_large_100 --batch-size 512 -j 48 --model tf_mobilenetv3_large_100 --teacher-model gluon_senet154,gluon_resnet152_v1s --imagenet [imagenet-folder with train and val folders]

# # 224 x 224 EfficientNet-B0
# python train.py --save MEAL_V2_efficientnet_b0 --batch-size 512 -j 48 --model tf_efficientnet_b0 --teacher-model gluon_senet154,gluon_resnet152_v1s --imagenet [imagenet-folder with train and val folders]

# 380 x 380 ResNet-50
# python train.py --save MEAL_V2_resnet50_380 --batch-size 512 -j 48 --model resnet50 --image-size 380 --teacher-model tf_efficientnet_b4_ns,tf_efficientnet_b4 --imagenet [imagenet-folder with train and val folders]
