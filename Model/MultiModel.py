import torch
from torch import nn
# 导入实现的module
from Model.TextModel import TextModel
from Model.ImageModel import ImageModel


class MultiModel(nn.Module):
    def __init__(self, args):
        super(MultiModel, self).__init__()

        # 进行多模态分类
        self.multi_modal_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(args['hidden_size']*2, args['middle_hidden_size']),
            nn.ReLU(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(args['middle_hidden_size'], 3)
        )

        # 进行单模态分类
        self.single_modal_classifier = nn.Sequential(
            nn.Linear(args['hidden_size'], args['middle_hidden_size']),
            nn.ReLU(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(args['middle_hidden_size'], 3)
        )
        #调用已实现的单模态模型
        self.text_model = TextModel(args)
        self.image_model = ImageModel(args)


    def forward(self, texts=None, imgs=None):
        #只有文本数据
        if texts is None:
            img_out = self.image_model(imgs)
            out = self.single_modal_classifier(img_out)
            return out
        #只有图像数据
        if imgs is None:
            text_out = self.text_model(texts)
            out = self.single_modal_classifier(text_out)
            return out

        # 多模态数据
        text_out = self.text_model(texts)  # N, E
        img_out = self.image_model(imgs)  # N, E
        
        # 直接将特征进行拼接来进行分类
        out = self.multi_modal_classifier(torch.cat((text_out, img_out), 1))

        return out