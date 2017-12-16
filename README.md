# XI LSTM


# INSTALL

Python3.6 is required.

## Python Dependencies

```shell
pip3 install -r requirements.txt 
```

## Data Files

Download all the files that we needed automatically.

```shell
make download
```

Those files include:
1. train_test清洗后的数据集，分为训练集和测试集  
    注意：train_y_real_filter、train_x_real_filter是train_y_real、train_x_real的清洗之后的版本
1. word2vec训练好的词向量

## VSCode Extensions

1. Python
1. EditorConfig

# RUN

```shell
make run
```

# MODEL

Embedding -> Dropout -> LSTMP（编码层） -> LSTMO（解码层）-> Fully Connected Layer -> Softmax

# TODO LIST

- [ ] 现状：查看了EMBEDDING部分应该没有问题，但是输入到LSTMP layer和LSTMO layer之后对每句话的输出分布极为相似，导致后面对每个单词的标签预测都是同一个值，因此LOSS不降。标准LSTM的LOSS也不降，且标准LSTM layer1对每句话的输出分布不同，到标准LSTM layer2对每句话的输出分布就很相似了。
    改进：为什么分布几乎一样呢？
- [ ] 可能还会出现难以解释的问题。。。TAT
