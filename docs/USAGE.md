# CycleGAN Usage Guide

<!--
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
-->

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your dataset as follows:

```
datasets/
└── your_dataset/
    ├── trainA/  # Domain A training images
    ├── trainB/  # Domain B training images
    ├── testA/   # Domain A test images
    └── testB/   # Domain B test images
```

### 3. Training

Basic training:
```bash
python train.py --dataroot ./datasets/your_dataset --name experiment_name
```

Advanced training with mixed precision:
```bash
python train_advanced.py --dataroot ./datasets/your_dataset --name experiment_name --mixed_precision
```

### 4. Testing

```bash
python test.py --dataroot ./datasets/your_dataset --name experiment_name
```

### 5. Evaluation

```bash
python evaluate.py --dataroot ./datasets/your_dataset --name experiment_name
```

### 6. Web Interface

```bash
cd web
python app.py
```

Then open http://localhost:5000 in your browser.

## Advanced Features

### Mixed Precision Training

Use `train_advanced.py` with `--mixed_precision` flag for faster training with less memory.

### Distributed Training

```bash
python train_advanced.py --distributed --world_size 4
```

### TensorBoard Monitoring

```bash
tensorboard --logdir=./logs
```

### Download Pre-trained Models

```bash
python scripts/download_pretrained.py horse2zebra
```

## Jupyter Notebook

Open `CycleGAN_Demo.ipynb` for interactive experimentation.

