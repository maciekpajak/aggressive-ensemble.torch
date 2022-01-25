from aggressive_ensemble.Ensemble import Ensemble
from aggressive_ensemble.Classifier import Classifier
import aggressive_ensemble.utils.transforms as t
import aggressive_ensemble.utils.augmentations as a
import pandas as pd
import torch
import copy

# Additional transformation
class SwitchRGBChannels:
    """Transformacja zamieniająca kanały RGB"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, polygon, labels = sample["image"], sample["polygon"], sample["labels"]

        x = False
        if random.randrange(0, 100) <= 100 * self.p:
            x = True
        y = bool(random.getrandbits(1))

        if x:
            image = np.array(image)
            if labels["color_red"] == 1:
                labels["color_red"] = 0.0
                if y:
                    # switch red channel with blue channel
                    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
                    labels["color_blue"] = 1.0
                else:
                    # switch red channel with green channel
                    image[:, :, [0, 1, 2]] = image[:, :, [1, 0, 2]]
                    labels["color_green"] = 1.0
            elif labels["color_green"] == 1:
                labels["color_green"] = 0.0
                if y:
                    # switch green channel with blue channel
                    image[:, :, [0, 1, 2]] = image[:, :, [0, 2, 1]]
                    labels["color_blue"] = 1.0
                else:
                    # switch green channel with red channel
                    image[:, :, [0, 1, 2]] = image[:, :, [1, 0, 2]]
                    labels["color_green"] = 1.0
            elif labels["color_blue"] == 1:
                labels["color_blue"] = 0.0
                if y:
                    # switch red channel with blue channel
                    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
                    labels["color_red"] = 1.0
                else:
                    # switch blue channel with green channel
                    image[:, :, [0, 1, 2]] = image[:, :, [0, 2, 1]]
                    labels["color_green"] = 1.0
            image = Image.fromarray(image)

        sample = {'image': image, 'polygon': polygon, 'labels': labels}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"

# ----------------------


root_dir = '/content/drive/My Drive/'
device = "gpu"
labels_csv = root_dir + 'praca/labels.csv'
data_dir = root_dir + 'data_cropped/data/'
save_dir = root_dir + 'save/'
train_csv = root_dir + 'data_cropped/train_cropped_he.csv'
val_csv = root_dir + 'data_cropped/val_cropped_he.csv'
test_csv = root_dir + 'data_cropped/test_cropped_he.csv'
train_df = pd.DataFrame(pd.read_csv(train_csv))
val_df = pd.DataFrame(pd.read_csv(val_csv))
test_df = pd.DataFrame(pd.read_csv(test_csv))
labels = list(pd.read_csv(labels_csv))

vgg = torch.load('/content/drive/My Drive/basic_models/vgg_pretrained.pth')

from torch import nn

vgg1 = Classifier(name="vgg_N__pretrained",
                  labels=labels,
                  model=copy.deepcopy(vgg),
                  device=device,
                  feature_extract=True,
                  is_inception=False,
                  preprocessing=[],
                  augmentation=[],
                  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                  input_size=224, lr=0.01, momentum=0.9, val_every=5,
                  save_every=5, shuffle=True, criterion=nn.BCELoss(),
                  checkpoint_path='/content/drive/My Drive/save/vgg_N__pretrained/bestfitness_from_validation.pth')

vgg2 = Classifier(name="vgg_PE+CC+N_RHF+RVF_pretrained",
                  labels=labels,
                  model=copy.deepcopy(vgg),
                  device=device,
                  feature_extract=True,
                  is_inception=False,
                  preprocessing=[t.ExtractPolygon(), t.ChangeColorspace("RGB", "HSV")],
                  augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                  input_size=224, lr=0.01, momentum=0.9, val_every=5,
                  save_every=5, shuffle=True, criterion=nn.BCELoss(),
                  checkpoint_path='/content/drive/My Drive/save/vgg_PE+CC+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

vgg3 = Classifier(name="vgg_PE+N_RHF+RVF_pretrained",
                  labels=labels,
                  model=copy.deepcopy(vgg),
                  device=device,
                  feature_extract=True,
                  is_inception=False,
                  preprocessing=[t.ExtractPolygon()],
                  augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                  input_size=224, lr=0.01, momentum=0.9, val_every=5,
                  save_every=5, shuffle=True, criterion=nn.BCELoss(),
                  checkpoint_path='/content/drive/My Drive/save/vgg_PE+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

vgg4 = Classifier(name="vgg_PE+N_RHF+RVF+RR_pretrained",
                  labels=labels,
                  model=copy.deepcopy(vgg),
                  device=device,
                  feature_extract=True,
                  is_inception=False,
                  preprocessing=[t.ExtractPolygon()],
                  augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip(), a.RandomRotate()],
                  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                  input_size=224, lr=0.01, momentum=0.9, val_every=5,
                  save_every=5, shuffle=True, criterion=nn.BCELoss(),
                  checkpoint_path='/content/drive/My Drive/save/vgg_PE+N_RHF+RVF+RR_pretrained/bestfitness_from_validation.pth')

vgg6 = Classifier(name="vgg_PE+RTH+N_RHF+RVF_pretrained",
                  labels=labels,
                  model=copy.deepcopy(vgg),
                  device=device,
                  feature_extract=True,
                  is_inception=False,
                  preprocessing=[t.ExtractPolygon(), t.RotateToHorizontal()],
                  augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                  input_size=224, lr=0.01, momentum=0.9, val_every=5,
                  save_every=5, shuffle=True, criterion=nn.BCELoss(),
                  checkpoint_path='/content/drive/My Drive/save/vgg_PE+RTH+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

vgg7 = Classifier(name="vgg_PE+N_RVF+RVF+SCH_pretrained",
                  labels=labels,
                  model=copy.deepcopy(vgg),
                  device=device,
                  feature_extract=True,
                  is_inception=False,
                  preprocessing=[t.ExtractPolygon()],
                  augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip(), SwitchRGBChannels()],
                  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                  batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                  input_size=224, lr=0.01, momentum=0.9, val_every=5,
                  save_every=5, shuffle=True, criterion=nn.BCELoss(),
                  checkpoint_path='/content/drive/My Drive/save/vgg_PE+N_RVF+RVF+SCH_pretrained/bestfitness_from_validation.pth')

alexnet1 = Classifier(name="alexnet_BCE_PE+N_RHF+RVF_pretrained",
                      labels=labels,
                      model=copy.deepcopy(alexnet),
                      device=device,
                      feature_extract=True,
                      is_inception=False,
                      preprocessing=[t.ExtractPolygon()],
                      augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                      batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                      input_size=224, lr=0.01, momentum=0.9, val_every=5,
                      save_every=5, shuffle=True, criterion=nn.BCELoss(),
                      checkpoint_path='/content/drive/My Drive/save/alexnet_PE+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

alexnet3 = Classifier(name="alexnet_PE+RTH+N_RHF+RVF_pretrained",
                      labels=labels,
                      model=copy.deepcopy(alexnet),
                      device=device,
                      feature_extract=True,
                      is_inception=False,
                      preprocessing=[t.ExtractPolygon(), t.RotateToHorizontal()],
                      augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                      batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                      input_size=224, lr=0.01, momentum=0.9, val_every=5,
                      save_every=5, shuffle=True, criterion=nn.BCELoss(),
                      checkpoint_path='/content/drive/My Drive/save/alexnet_PE+RTH+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

alexnet4 = Classifier(name="alexnet_PE+N_RVF+RVF+SCH_pretrained",
                      labels=labels,
                      model=copy.deepcopy(alexnet),
                      device=device,
                      feature_extract=True,
                      is_inception=False,
                      preprocessing=[t.ExtractPolygon()],
                      augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip(), SwitchRGBChannels()],
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                      batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                      input_size=224, lr=0.01, momentum=0.9, val_every=5,
                      save_every=5, shuffle=True, criterion=nn.BCELoss(),
                      checkpoint_path='/content/drive/My Drive/save/alexnet_PE+N_RVF+RVF+SCH_pretrained/bestfitness_from_validation.pth')

nasnetmobile1 = Classifier(name="nasnetmobile_PE+N_RHF+RVF_pretrained",
                           labels=labels,
                           model=copy.deepcopy(nasnetmobile),
                           device=device,
                           feature_extract=True,
                           is_inception=False,
                           preprocessing=[t.ExtractPolygon()],
                           augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                           mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                           batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                           input_size=224, lr=0.01, momentum=0.9, val_every=5,
                           save_every=5, shuffle=True, criterion=nn.BCELoss(),
                           checkpoint_path='/content/drive/My Drive/save/nasnetmobile_PE+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

nasnetmobile2 = Classifier(name="nasnetmobile_PE+RTH+N_RHF+RVF_pretrained",
                           labels=labels,
                           model=copy.deepcopy(nasnetmobile),
                           device=device,
                           feature_extract=True,
                           is_inception=False,
                           preprocessing=[t.ExtractPolygon(), t.RotateToHorizontal()],
                           augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                           mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                           batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                           input_size=224, lr=0.01, momentum=0.9, val_every=5,
                           save_every=5, shuffle=True, criterion=nn.BCELoss(),
                           checkpoint_path='/content/drive/My Drive/save/nasnetmobile_PE+RTH+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

xception1 = Classifier(name="xception_PE+N_RHF+RVF_pretrained",
                       labels=labels,
                       model=xception,
                       device=copy.deepcopy(device),
                       feature_extract=True,
                       is_inception=False,
                       preprocessing=[t.ExtractPolygon()],
                       augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                       mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                       batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                       input_size=299, lr=0.01, momentum=0.9, val_every=5,
                       save_every=5, shuffle=True, criterion=nn.BCELoss(),
                       checkpoint_path='/content/drive/My Drive/save/xception_PE+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

nasnetmobile4 = Classifier(name="nasnetmobile_PE+N_RHF+RVF+RR_pretrained",
                           labels=labels,
                           model=copy.deepcopy(nasnetmobile),
                           device=device,
                           feature_extract=True,
                           is_inception=False,
                           preprocessing=[t.ExtractPolygon(), t.RotateToHorizontal()],
                           augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip(), a.RandomRotate()],
                           mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                           batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                           input_size=224, lr=0.01, momentum=0.9, val_every=5,
                           save_every=5, shuffle=True, criterion=nn.BCELoss(),
                           checkpoint_path='/content/drive/My Drive/save/nasnetmobile_PE+N_RHF+RVF+RR_pretrained/bestfitness_from_validation.pth')

inception1 = Classifier(name="inception_N__pretrained",
                        labels=labels,
                        model=copy.deepcopy(inception),
                        device=device,
                        feature_extract=True,
                        is_inception=False,
                        preprocessing=[],
                        augmentation=[],
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                        batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                        input_size=299, lr=0.01, momentum=0.9, val_every=5,
                        save_every=5, shuffle=True, criterion=nn.BCELoss(),
                        checkpoint_path='/content/drive/My Drive/save/inception_N__pretrained/bestfitness_from_validation.pth')

inception2 = Classifier(name="inception_PE+RTH+N_RHF+RVF_pretrained",
                        labels=labels,
                        model=copy.deepcopy(inception),
                        device=device,
                        feature_extract=True,
                        is_inception=False,
                        preprocessing=[t.ExtractPolygon(), t.RotateToHorizontal()],
                        augmentation=[a.RandomHorizontalFlip(), a.RandomVerticalFlip()],
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                        batch_size=50, num_workers=2, epochs=1, start_epoch=0,
                        input_size=299, lr=0.01, momentum=0.9, val_every=5,
                        save_every=5, shuffle=True, criterion=nn.BCELoss(),
                        checkpoint_path='/content/drive/My Drive/save/inception_PE+RTH+N_RHF+RVF_pretrained/bestfitness_from_validation.pth')

ensemble_structure = {
    "subensemble1": {
        "labels": ['sunroof', 'luggage_carrier', 'open_cargo_area', 'enclosed_cab', 'spare_wheel',
                   'wrecked', 'flatbed', 'ladder', 'enclosed_box', 'soft_shell_box', 'harnessed_to_a_cart',
                   'ac_vents'],
        "classifiers": [vgg7, vgg3, vgg6, vgg1, alexnet4,
                        alexnet1, vgg4, vgg2, nasnetmobile2, inception1]
    },
    "subensemble2": {
        "labels": ['general_class_large vehicle', 'general_class_small vehicle'],
        "classifiers": [vgg3,
                        vgg6,
                        vgg2,
                        nasnetmobile4,
                        vgg7,
                        alexnet1,
                        vgg4,
                        inception2,
                        xception1,
                        nasnetmobile2],
    },
    "subensemble3": {
        "labels": ['sub_class_bus', 'sub_class_cement mixer', 'sub_class_crane truck',
                   'sub_class_dedicated agricultural vehicle', 'sub_class_hatchback',
                   'sub_class_jeep', 'sub_class_light truck', 'sub_class_minibus',
                   'sub_class_minivan', 'sub_class_pickup', 'sub_class_prime mover',
                   'sub_class_sedan', 'sub_class_tanker', 'sub_class_truck', 'sub_class_van'],
        "classifiers": [vgg3,
                        vgg6,
                        vgg7,
                        vgg2,
                        alexnet4,
                        alexnet1,
                        nasnetmobile1,
                        alexnet3,
                        vgg1,
                        vgg4]
    },
    "subensemble4": {
        "labels": ['color_black', 'color_blue', 'color_green', 'color_other', 'color_red', 'color_silver/grey',
                   'color_white', 'color_yellow'],
        "classifiers": [vgg3,
                        vgg4,
                        vgg6,
                        alexnet1,
                        vgg1,
                        alexnet3,
                        inception1,
                        vgg2,
                        inception2,
                        nasnetmobile1]
    }
}

ensemble = Ensemble(id='example_ensemble',
                    save_dir=save_dir,
                    labels=labels,
                    ensemble_structure=ensemble_structure,
                    device="gpu")

# Output
# Outputs will be saved in /content/drive/My Drive/save/example_ensemble/


answer_probabilities, answer_01, answer_ranking = ensemble(data_dir=data_dir, test_df=test_df)

# Output:
# Detecting...
# ensemble211209_test7
# ├╴subensemble1
# │ └╴vgg_PE + N_RVF + RVF + SCH_pretrained
# Detecting: 100 % |██████████ | 117 / 117[13:01 < 00:00, 6.68
# s / it]
# │ └╴vgg_PE + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[00:53 < 00:00, 2.18
# it / s]
# │ └╴vgg_PE + RTH + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[01:03 < 00:00, 1.85
# it / s]
# │ └╴vgg_N__pretrained
# Detecting: 100 % |██████████ | 117 / 117[00:46 < 00:00, 2.53
# it / s]
# │ └╴alexnet_PE + N_RVF + RVF + SCH_pretrained
# Detecting: 100 % |██████████ | 117 / 117[00:42 < 00:00, 2.75
# it / s]
# │ └╴alexnet_BCE_PE + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[00:42 < 00:00, 2.78
# it / s]
# │ └╴vgg_PE + N_RHF + RVF + RR_pretrained
# Detecting: 100 % |██████████ | 117 / 117[00:53 < 00:00, 2.20
# it / s]
# │ └╴vgg_PE + CC + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[00:55 < 00:00, 2.11
# it / s]
# │ └╴nasnetmobile_PE + RTH + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[01:03 < 00:00, 1.84
# it / s]
# │ └╴inception_N__pretrained
# Detecting: 100 % |██████████ | 117 / 117[01:01 < 00:00, 1.91
# it / s]
# -- merging
# subensemble1
# ├╴subensemble2
# │ └╴vgg_PE + N_RHF + RVF_pretrained
# │ └╴vgg_PE + RTH + N_RHF + RVF_pretrained
# │ └╴vgg_PE + CC + N_RHF + RVF_pretrained
# │ └╴nasnetmobile_PE + N_RHF + RVF + RR_pretrained
# Detecting: 100 % |██████████ | 117 / 117[01:02 < 00:00, 1.86
# it / s]
# │ └╴vgg_PE + N_RVF + RVF + SCH_pretrained
# │ └╴alexnet_BCE_PE + N_RHF + RVF_pretrained
# │ └╴vgg_PE + N_RHF + RVF + RR_pretrained
# │ └╴inception_PE + RTH + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[01:13 < 00:00, 1.60
# it / s]
# │ └╴xception_PE + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[01:56 < 00:00, 1.01
# it / s]
# │ └╴nasnetmobile_PE + RTH + N_RHF + RVF_pretrained
# -- merging
# subensemble2
# ├╴subensemble3
# │ └╴vgg_PE + N_RHF + RVF_pretrained
# │ └╴vgg_PE + RTH + N_RHF + RVF_pretrained
# │ └╴vgg_PE + N_RVF + RVF + SCH_pretrained
# │ └╴vgg_PE + CC + N_RHF + RVF_pretrained
# │ └╴alexnet_PE + N_RVF + RVF + SCH_pretrained
# │ └╴alexnet_BCE_PE + N_RHF + RVF_pretrained
# │ └╴nasnetmobile_PE + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[00:53 < 00:00, 2.18
# it / s]
# │ └╴alexnet_PE + RTH + N_RHF + RVF_pretrained
# Detecting: 100 % |██████████ | 117 / 117[00:51 < 00:00, 2.26
# it / s]
# │ └╴vgg_N__pretrained
# │ └╴vgg_PE + N_RHF + RVF + RR_pretrained
# -- merging
# subensemble3
# ├╴subensemble4
# │ └╴vgg_PE + N_RHF + RVF_pretrained
# │ └╴vgg_PE + N_RHF + RVF + RR_pretrained
# │ └╴vgg_PE + RTH + N_RHF + RVF_pretrained
# │ └╴alexnet_BCE_PE + N_RHF + RVF_pretrained
# │ └╴vgg_N__pretrained
# │ └╴alexnet_PE + RTH + N_RHF + RVF_pretrained
# │ └╴inception_N__pretrained
# │ └╴vgg_PE + CC + N_RHF + RVF_pretrained
# │ └╴inception_PE + RTH + N_RHF + RVF_pretrained
# │ └╴nasnetmobile_PE + N_RHF + RVF_pretrained
# -- merging
# subensemble4




