！！！代码在master分支

# 实验五——多模态情感分类

## 环境配置

环境配置在文件 requirements.txt 中，请使用该命令查看

```py
pip install -r requirements.txt
```

## 项目框架

```py


|-- data_processing.py # 预处理文件
|-- data_process.py # 预处理文件
|-- model.pth  #存放模型的文件
|-- datasets     #存放数据的文件夹
    |-- data #原始数据
    |-- dev.json  #处理后的验证集
    |-- test_without_label.txt #测试文件
    |-- train.txt #训练数据
    |-- test.json # 处理后的测试集
    |-- train.json #处理后的训练集
|-- Model # 所有模型
    |-- ImageModel.py  #图像模型
    |-- TextModel.py   #文本模型
    |-- MultiModel.py #多模态模型
|-- train.py # 训练、验证、预测函数
|-- main.py # 主函数
|-- requirements.txt # 环境依赖
|-- test_with_label.txt #预测文件
|-- README.md #README
```



## 运行方法

先配置环境依赖：

```py
pip install -r requirements.txt
```

生成.json文件
```py
python data_processing.py
```

当出现了.json文件即可开始运行main.py
此时是多模态模型结果0.63
  
## 消融实验

消融实验运行需要修改train.py函数中的choose_model默认参数，再运行main.py
图像模型：0.60
文本模型：0.59

##参考文件

https://github.com/google/sentencepiece
https://github.com/RecklessRonan/GloGNN/blob/master/readme.md
