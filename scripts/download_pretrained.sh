#!/bin/bash
set -e
mkdir -p pretrained
cd pretrained
wget https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth
echo done
