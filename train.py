from cProfile import label
import torch
from torch import nn
from Model.MultiModel import MultiModel

# 定义模型、分类器、损失函数、epoch
args = {
    'hidden_size': 1000,
    'middle_hidden_size': 500,
    'lr': 1e-5,  
    'epochs': 3,
    'dropout': 0.0,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#PyTorch 模型只能在指定的设备上运行
model = MultiModel(args).to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_function = nn.CrossEntropyLoss()
epochs = 3


#开始训练
def train(model,train_dataloader,dev_dataloader, choose_model='multi'):
    
    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        train_acc_total,train_loss_total = 0.0, 0.0
        total = 0
        for step, batch_datas in enumerate(train_dataloader):
            _, texts, imgs, label = batch_datas
            texts = texts.to(device=device)
            imgs = imgs.to(device=device)
            label = label.to(device=device)
            total += label.shape[0]
            
            # 训练模型
            if choose_model == 'multi':
                output = model(texts=texts, imgs=imgs)
            elif choose_model == 'text':
                output = model(texts=texts, imgs=None)
            elif choose_model == 'img':
                output = model(texts=None, imgs=imgs)

            # 计算损失
            loss = loss_function(output, label.long()).sum()

            # 计算train的loss和acc
            train_loss_total += loss.item()
            train_acc_total += (output.argmax(dim=1) == label).float().sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 输出训练集准确率
        print('epoch %d, loss %.4f, train acc %.4f' % (epoch, train_loss_total / total, train_acc_total / total))  
        
        # 输出验证集准确率
        acc = validation(model, dev_dataloader, epoch, choose_model='multi')
        
        #保存模型
        if best_accuracy < acc:
            best_accuracy = acc
            torch.save(model, 'model.pth')

#开始验证   
def validation(model, dev_dataloader, epoch=None, choose_model='multi'):
    # 计算验证集准确率
    acc = 0.0
    total = 0
    for step, batch_datas in enumerate(dev_dataloader):
        ids, texts, imgs, label = batch_datas
        texts = texts.to(device=device)
        imgs = imgs.to(device=device)
        label  = label .to(device=device)
        total += label.shape[0]
        # 选择模型
        if choose_model == 'multi':
            output = model(texts=texts, imgs=imgs)
        elif choose_model == 'text':
            output = model(texts=texts, imgs=None)
        elif choose_model == 'img':
            output = model(texts=None, imgs=imgs)
        # 准确率
        acc += (output.argmax(dim=1) == label).float().sum().item()
        
    print('epoch %d, dev acc %.4f'% (epoch, acc / total))
    return acc / total
    
#开始预测
def predict(model,test_dataloader):
    # 加载保存的模型
    model = torch.load('model.pth')

    predict_list = []
    for step, batch_datas in enumerate(test_dataloader):
        ids, texts, imgs, y = batch_datas
        texts = texts.to(device=device)
        imgs = imgs.to(device=device)
        muli_label = model(texts=texts, imgs=imgs)
        predict_y = muli_label.argmax(dim=1)  # 使用主分类器
        for i in range(len(ids)):
            item_id = ids[i]
            # 这里是将数字转换成标签
            tag = test_dataloader.dataset.map2[int(predict_y[i])]
            predict_dict = {
                'guid': item_id,
                'tag': tag,
            }
            predict_list.append(predict_dict)

    # 保存预测结果
    with open('./datasets/test.json', 'w', encoding='utf-8') as f:
        f.write('guid,tag\n')
        for pred in predict_list:
            f.write(f"{pred['guid']},{pred['tag']}\n")

