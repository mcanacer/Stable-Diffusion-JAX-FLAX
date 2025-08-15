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
        text_model=None,
        text_drop_prob=0.0,
        max_length=77
    ):
        super(CocoDataset, self).__init__()

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.text_drop_prob = text_drop_prob
        self.max_length = max_length

        if self.tokenizer is not None:
            token = self.tokenizer(
                '',
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )

            indexed_tokens = token['input_ids']
            att_masks = token['attention_mask']

            indexed_tokens = torch.tensor(indexed_tokens)
            att_masks = torch.tensor(att_masks)

            self.empty_text_embed = text_model(indexed_tokens, att_masks).last_hidden_state

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
                return_attention_mask=True,
            )

            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            text_embed = self.text_model(
                input_ids,
                attention_mask,
            ).last_hidden_state

            if self.text_drop_prob > 0:
                text_drop_mask = torch.zeros((image.shape[0])).float().uniform(0, 1) < self.text_drop_prob
                text_embed[text_drop_mask, ...] = self.empty_text_embed[0]

            return image, text_embed
        else:
            return image
