#!/usr/bin/env bash
set -e

cd data

if [ -d train_test ]
then
    echo 'train_test already exists'
else
    curl -LO https://github.com/zixia/xi-lstm/releases/download/v0.0.1/train_test.zip
    unzip train_test.zip
    rm train_test.zip
fi

if [ -d word2vec ]
then
    echo 'word2vec already exists'
else
    curl -LO https://github.com/zixia/xi-lstm/releases/download/v0.0.1/word2vec.zip
    unzip word2vec.zip
    rm word2vec.zip
fi

if [ -f word2vec_google300_for_NYT.pkl ]
then
    echo 'word2vec_google already exists'
else
    curl -LO https://github.com/zixia/xi-lstm/releases/download/v0.0.1/word2vec_google300_for_NYT.pkl
fi

if [ -d multi_label/train_test ]
then
    echo 'multi_label/train_test exists'
else
    curl -LO https://github.com/zixia/xi-lstm/releases/download/v0.0.1/multi_label.zip
    unzip multi_label.zip
    rm multi_label.zip
fi
