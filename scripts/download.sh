#!/usr/bin/env bash
set -e

cd data
if [ ! -d train_test ]; then curl -LO https://github.com/zixia/xi-lstm/releases/download/v0.0.1/train_test.zip && unzip train_test.zip && rm train_test.zip; fi
if [ ! -d word2vec ]; 	then curl -LO https://github.com/zixia/xi-lstm/releases/download/v0.0.1/word2vec.zip && unzip word2vec.zip && rm word2vec.zip; 	fi
if [ ! -f word2vec_google300_for_NYT.pkl ]; then curl -LO https://github.com/zixia/xi-lstm/releases/download/v0.0.1/word2vec_google300_for_NYT.pkl; fi
cd -
