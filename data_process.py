import numpy as np
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms #用于常见的图形变换
from transformers import XLMRobertaTokenizer
from torch.utils.data import DataLoader


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', use_fast=True)
        self.map1 = {'positive': 0, 'negative': 1, 'neutral': 2}
        self.map2 = {0:'positive',1:'negative',2:'neutral'}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        item = self.data[item]
        item_id = item['id']
        text = item['text']
        img = item['img']
        tag = item.get('tag', 'unknown')

        input_text = self.tokenizer(text, max_length=64, padding='max_length', return_tensors="pt", truncation=True)
        input_text['attention_mask'] = input_text['attention_mask'].squeeze()
        input_text['input_ids'] = input_text['input_ids'].squeeze()

        img = Image.fromarray(np.uint8(img))
        img = transforms.Resize((224, 224))(img)
        img = transforms.CenterCrop((224, 224))(img)
        if np.random.rand() > 0.5:
            img = transforms.RandomHorizontalFlip(p=1.0)(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        input_tag = self.map1.get(tag, 3)

        return item_id, input_text, img, input_tag

def load_data(file, shuffle=True):
    data_list = []
    with open(file, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for _, item in enumerate(items):
            data_item = {}
            data_item['id'] = item['guid']
            data_item['text'] = item['text']
            data_item['img'] = np.array(Image.open(item['img']))
            data_item['tag'] = item.get('tag', 'unknown')
            data_list.append(data_item)
    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=4)
    return dataloader