import torch
import torch.utils.data
import torchvision

import json

from PIL import Image

class MNISTDetection(object):
    def __init__(self, img_folder, ann_file, transforms):
        super(MNISTDetection, self).__init__()
        self._transforms = transforms
        self._images = img_folder
        self._targets = json.load(open(ann_file))
        self._ids = list((self._targets.keys()))

    def __getitem__(self, idx):
        image_id = self._ids[idx]
        u_targets = self._targets[image_id]
        u_img = Image.open(f'{self._images}/{image_id}.png').convert('RGB')

        target = {}
        target['labels'] = torch.DoubleTensor(list(map(lambda x: x['target'], u_targets)))
        target['boxes'] = torch.DoubleTensor(list(map(lambda x: x['bbox'], u_targets)))
        target['image_id'] = torch.DoubleTensor([int(image_id)])
        target['orig_size'] = torch.DoubleTensor(u_img.size)
        target['size'] = torch.DoubleTensor(u_img.size)


        if self._transforms is not None:
            img, target = self._transforms(u_img, target)

        return img, target