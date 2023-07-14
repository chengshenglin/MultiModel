import torch
from torch import nn

from data_process import load_data
from train import train,predict
from Model.MultiModel import MultiModel


args = {
    'hidden_size': 1000,
    'middle_hidden_size': 500,
    'lr': 1e-5,  
    'dropout': 0.0,
}
if __name__ == '__main__':
    
    # 加载数据集
    train_dataloader = load_data('./datasets/train.json', shuffle=True)
    dev_dataloader = load_data('./datasets/dev.json', shuffle=True)
    test_dataloader = load_data('./datasets/test.json', shuffle=False)

    # 定义模型、分类器、损失函数、epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModel(args).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_function = nn.CrossEntropyLoss()
    epochs = 3

    print('training...')
    train(model,train_dataloader,dev_dataloader, choose_model='multi')
    print('predicting...')
    predict(model,test_dataloader)
    
    
