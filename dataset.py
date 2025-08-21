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
        max_length=77,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.text_drop_prob = float(text_drop_prob)
        self.max_length = max_length
        self.image_transforms = transform

        # Precompute empty text embedding
        if self.tokenizer is not None and self.text_model is not None:
            self.text_model.eval()
            with torch.no_grad():
                tokens = self.tokenizer(
                    "",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                self.empty_text_embed = self.text_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                ).last_hidden_state  # [1, max_len, hidden]

        self.coco = COCO(annotation_file_path)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.dataset_dir, path)

        image = Image.open(img_path).convert("RGB")
        if self.image_transforms:
            image = self.image_transforms(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = self.coco.loadAnns(ann_ids)
        caption = random.choice(captions)["caption"]

        if self.tokenizer is None or self.text_model is None:
            return image

        with torch.no_grad():
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_embed = self.text_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            ).last_hidden_state  # [1, max_len, hidden]

            # dropout
            if random.random() < self.text_drop_prob:
                text_embed = self.empty_text_embed.clone()

        return image, text_embed.squeeze(0)  # [max_len, hidden]
