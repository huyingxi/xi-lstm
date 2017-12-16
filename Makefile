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
download:
	scripts/download.sh

.PHONY: test
test: check-version lint
	echo "test"
	# false
	echo "ok"

.PHONY: check-version
check-version:
	./scripts/check_version.py

code:
	# code src/	# vscode need to use src as root dir
	PYTHONPATH=src/ code .

run:
	python3 src/main.py

run-standard:
	python3 src/main_standard_lstm.py

huan:
	python3 src/huan.py
