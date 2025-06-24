<div align="center">

# Boosting Medical Image Synthesis via Registration-guided Consistency and Disentanglement Learning

![overview](pic/overview.jpg)

</div>

## Description

This is the source code for MICCAI paper (ID: 2212) - "Boosting Medical Image Synthesis via Registration-guided Consistency and Disentanglement Learning"

## Requirements

* python 3.8
* pytorch 1.10
* tensorboardX
* SimpleITK

## Model training

```bash
# train on GPU
python train.py --image_dir=/path/to/your/data --max_epochs=200 --lr_max=0.0002
```

## Model evaluation

```bash
python test.py --image_dir=/path/to/your/data --checkpoint_dir=/path/to/your/model_results 
```

