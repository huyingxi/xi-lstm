# Makefile for xi-lstm
# Author: Huan LI <zixia@zixia.net> git.io/zixia

SOURCE_GLOB=$(wildcard src/*.py)

.PHONY: all
all : clean lint

.PHONY: clean
clean:
	echo "TODO: clean what?"

.PHONY: lint
lint: pylint pep8 flake8 mypy

.PHONY: pylint
pylint:
	pylint $(SOURCE_GLOB)

.PHONY: pep8
pep8:
	pep8 $(SOURCE_GLOB)

.PHONY: flake8
flake8:
	flake8 $(SOURCE_GLOB)

.PHONY: mypy
mypy:
	MYPYPATH=stubs/ mypy \
		--disallow-untyped-defs \
		--python-version=3.6 \
		$(SOURCE_GLOB)

.PHONY: download
.ONESHELL:
download:
	# https://stackoverflow.com/a/30590240/1123955
	cd datasets
	if [ ! -d train_test ]; then wget https://github.com/zixia/xi-lstm/releases/download/v0.0.1/train_test.zip && unzip train_test.zip && rm train_test.zip; fi
	if [ ! -d word2vec ]; 	then wget https://github.com/zixia/xi-lstm/releases/download/v0.0.1/word2vec.zip && unzip word2vec.zip && rm word2vec.zip; 	fi
	cd -

.PHONY: test
test: check-version lint
	echo "test"
	# false
	echo "ok"

.PHONY: check-version
check-version:
	./util/check_version.py

code:
	# code src/	# vscode need to use src as root dir
	PYTHONPATH=src/ code .
