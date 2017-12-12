# XI LSTM

1. LSTMPcell.py文件是模型的训练和测试文件
1. modules.zip文件是pytorch被修改的部分，需要替换掉原本的torch/nn/modules
1. train_test是清洗后的数据集，分为训练集和测试集，需要修改LSTMPcell里的调用路径，注意：train_y_real_filter、train_x_real_filter是train_y_real、train_x_real的清洗之后的版本
1. word2vec是训练好的词向量，解压出来修改LSTMPcell中mode.bin的调用路径即可
1. 现在的模型是
Embedding -> Dropout -> LSTMP（编码层） -> LSTMO（解码层）-> Fully Connected Layer -> Softmax
1. 需要的改进
    1. 现状：LSTMP（编码层）应该是双向的，而且我在模型中的bidirectional = True进行了设置，但是现在还是单向的  
    改进：为什么还是单向的呢！！？？
	1. 现状：每个句子的长度不同，现在取最大值188个单词进行padding，padding为‘O’；比如：一个句子本来只有40个单词，padding到188个单词，所以后面（还是前面？）会多148个占为标签‘O’，那么LSTM的输入也是（batch_size, 188, 300），输出也是(batch_size, 188, 300)  
    改进：能够在LSTMP和LSTMO中实现对padding‘O’的过滤，即：能不能将padding的无效标签在通过LSTM自动删除，将上述40(N)个单词的输入通过LSTM之后变成（batch_size，40(N)，300）,N可变
    1. 现状：LOSS为什么不变呢？  
	改进：让它降低吧！
1. 可能还会出现难以解释的问题。。。TAT

# INSTALL

## Python Dependencies

```shell
pip3 install pylint
pip3 install nltk gensim
```

## Datasets

```shell
make download
```

## VSCode Extensions

1. Python
1. EditorConfig

