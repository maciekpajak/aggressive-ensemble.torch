from aggressive_ensemble.Classifier import Classifier
from aggressive_ensemble.utils.preprocessing import *
from aggressive_ensemble.utils.augmentations import *




name = "vgg_RTH+PE+N_RVF+RHF_pretrained"

import pandas as pd
root_dir = '/content/drive/My Drive/'
device = "gpu"
labels_csv = root_dir + 'labels.csv'
data_dir = root_dir + 'data/'
save_dir = root_dir + 'save/' + name + "/"
train_csv = root_dir + 'data_cropped/train.csv'
val_csv = root_dir + 'data_cropped/val.csv'
test_csv = root_dir + 'data_cropped/test.csv'
train_df = pd.DataFrame(pd.read_csv(train_csv))
val_df = pd.DataFrame(pd.read_csv(val_csv))
test_df = pd.DataFrame(pd.read_csv(test_csv))
labels = list(pd.read_csv(labels_csv))


import os
os.mkdir(save_dir)

import torch
model_path = '/content/drive/My Drive/basic_models/vgg_pretrained.pth'
model = torch.load(model_path)

from torch import nn
classifier = Classifier(name=name,
                        labels=labels,
                        model=model,
                        device="gpu",
                        feature_extract=True,
                        is_inception=False,
                        preprocessing=[RotateToHorizontal(), ExtractPolygon()],
                        augmentation=[RandomVerticalFlip(),RandomHorizontalFlip()],
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        batch_size=32,
                        num_workers=2,
                        epochs=50,
                        start_epoch=0,
                        input_size=224,
                        lr=0.01, momentum=0.9,
                        val_every=2,
                        save_every=5,
                        shuffle=True,
                        criterion=nn.BCELoss())
print(classifier)

# Output:
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (6): ReLU(inplace=True)
#     (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (8): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (10): ReLU(inplace=True)
#     (11): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (13): ReLU(inplace=True)
#     (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (16): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (17): ReLU(inplace=True)
#     (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (19): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (20): ReLU(inplace=True)
#     (21): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (23): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (24): ReLU(inplace=True)
#     (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (26): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (27): ReLU(inplace=True)
#     (28): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Sequential(
#       (0): Linear(in_features=4096, out_features=37, bias=True)
#       (1): Sigmoid()
#     )
#   )
# )

from aggressive_ensemble.utils.metrics import mAP_score
model, train_stats, val_stats, _ = classifier.train(train_df=train_df, val_df=val_df, score_function=mAP_score,data_dir=data_dir, save_dir=save_dir)

# Output:
#      Epoch      loss     score
#
#       1/50    0.1819    0.1349: 100%|██████████| 146/146 [14:22<00:00,  5.91s/it
#       2/50    0.1239    0.2733: 100%|██████████| 146/146 [00:55<00:00,  2.61it/s
#        Val    0.1176    0.3393: 100%|██████████| 37/37 [03:20<00:00,  5.43s/it]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
#
#       3/50    0.1109     0.323: 100%|██████████| 146/146 [00:56<00:00,  2.56it/s
#       4/50     0.103    0.3621: 100%|██████████| 146/146 [00:57<00:00,  2.56it/s
#        Val    0.1034    0.4131: 100%|██████████| 37/37 [00:10<00:00,  3.44it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
#
#       5/50   0.09793    0.3868: 100%|██████████| 146/146 [01:00<00:00,  2.43it/s
#
# Autosave... saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/autosave.pth
#
#       6/50    0.0929    0.4151: 100%|██████████| 146/146 [00:55<00:00,  2.64it/s
#        Val   0.09841    0.4446: 100%|██████████| 37/37 [00:09<00:00,  4.05it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
#
#       7/50   0.08912    0.4334: 100%|██████████| 146/146 [00:54<00:00,  2.69it/s
#       8/50   0.08514    0.4742: 100%|██████████| 146/146 [00:53<00:00,  2.73it/s
#        Val   0.09175    0.4821: 100%|██████████| 37/37 [00:09<00:00,  4.03it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
#
#       9/50   0.08305    0.5007: 100%|██████████| 146/146 [00:54<00:00,  2.70it/s
#      10/50   0.07934    0.5134: 100%|██████████| 146/146 [00:52<00:00,  2.76it/s
#        Val   0.09115     0.499: 100%|██████████| 37/37 [00:08<00:00,  4.15it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
# Autosave... saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/autosave.pth
#
#      11/50   0.07666    0.5269: 100%|██████████| 146/146 [00:56<00:00,  2.60it/s
#      12/50   0.07433    0.5452: 100%|██████████| 146/146 [00:52<00:00,  2.78it/s
#        Val   0.08897    0.5717: 100%|██████████| 37/37 [00:08<00:00,  4.14it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
#
#      13/50   0.07208    0.5878: 100%|██████████| 146/146 [00:53<00:00,  2.72it/s
#      14/50    0.0716    0.5707: 100%|██████████| 146/146 [00:52<00:00,  2.77it/s
#        Val   0.08815    0.5807: 100%|██████████| 37/37 [00:08<00:00,  4.18it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
#
#      15/50   0.06724    0.6011: 100%|██████████| 146/146 [00:53<00:00,  2.74it/s
#
# Autosave... saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/autosave.pth
#
#      16/50   0.06555      0.59: 100%|██████████| 146/146 [00:54<00:00,  2.67it/s
#        Val   0.09012    0.5635: 100%|██████████| 37/37 [00:08<00:00,  4.19it/s]
#      17/50   0.06332    0.6173: 100%|██████████| 146/146 [00:52<00:00,  2.77it/s
#      18/50   0.06331    0.6239: 100%|██████████| 146/146 [00:52<00:00,  2.78it/s
#        Val   0.08886    0.5928: 100%|██████████| 37/37 [00:08<00:00,  4.16it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
#
#      19/50   0.06285    0.6444: 100%|██████████| 146/146 [00:53<00:00,  2.72it/s
#      20/50   0.05971    0.6612: 100%|██████████| 146/146 [00:52<00:00,  2.80it/s
#        Val    0.0897    0.5935: 100%|██████████| 37/37 [00:08<00:00,  4.14it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
# Autosave... saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/autosave.pth
#
#      21/50   0.05806    0.6719: 100%|██████████| 146/146 [00:56<00:00,  2.58it/s
#      22/50   0.05808    0.6623: 100%|██████████| 146/146 [00:51<00:00,  2.82it/s
#        Val   0.08977    0.6015: 100%|██████████| 37/37 [00:09<00:00,  4.08it/s]
#
# New best fitness! Saved in /content/drive/My Drive/save/vgg_RTH+PE+N_RVF+RHF_pretrained/bestfitness_from_validation.pth
# Early stopping! No progress
# Training complete in 39m 4s

answer = classifier(test_df=test_df, data_dir=data_dir, save_dir=save_dir)

# Output:
# Detecting: 100%|██████████| 182/182 [16:20<00:00,  5.39s/it]