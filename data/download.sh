#!/bin/bash
# 18GB Main Dataset with images
curl https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip --output ./data/CLEVR_v1.0.zip
unzip ./data/CLEVR_v1.0.zip -d data/
# 24GB Compositional Generalization Test wit images
# wget https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip
# unzip CLEVR_CoGenT_v1.0.zip
