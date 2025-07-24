import torch
from pycocotools.coco import COCO
from PIL import Image
import os
import random


class HuggingFace(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class CocoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        annotation_file_path,
        transform=None,
        tokenizer=None,
        max_length=77
    ):
        super(CocoDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.coco = COCO(annotation_file_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.image_transforms = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        path = coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.dataset_dir, path)
        image = Image.open(img_path).convert('RGB')
        image = self.image_transforms(image)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        captions = coco.loadAnns(ann_ids)
        caption = random.choice(captions)['caption']

        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = tokens.input_ids.squeeze(0)
            attention_mask = tokens.attention_mask.squeeze(0)
            return image, input_ids, attention_mask
        else:
            return image, caption
