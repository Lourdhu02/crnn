#!/bin/bash
set -e
if [ ! -d data/train ]; then mkdir -p data/train; fi
if [ ! -d data/val ]; then mkdir -p data/val; fi
if [ ! -d data/test ]; then mkdir -p data/test; fi
if [ ! -f data/train_labels.txt ]; then touch data/train_labels.txt; fi
if [ ! -f data/val_labels.txt ]; then touch data/val_labels.txt; fi
if [ ! -f data/test_labels.txt ]; then touch data/test_labels.txt; fi
echo ready
