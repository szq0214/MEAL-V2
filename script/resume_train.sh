# an example
python train.py --save MEAL_V2_resnet50_224 --batch-size 512 --model resnet50 --start-epoch 96 --teacher-model gluon_senet154,gluon_resnet152_v1s --student-state-file ./MEAL_V2_resnet50/model_state_95.pytar --imagenet [imagenet-folder with train and val folders] -j 40
