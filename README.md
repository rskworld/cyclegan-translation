# CycleGAN for Image-to-Image Translation

<!--
Project: CycleGAN for Image-to-Image Translation
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
Description: CycleGAN for unpaired image-to-image translation using cycle-consistent adversarial networks
-->

This project implements CycleGAN for image-to-image translation without paired training examples. The architecture uses two generators and two discriminators with cycle consistency loss to learn mappings between two image domains.

## Features

### Core Features
- **CycleGAN architecture**: Dual generator-discriminator setup
- **Unpaired image translation**: No need for paired training data
- **Cycle consistency loss**: Ensures meaningful translations
- **Style transfer capabilities**: Transform images between domains
- **Multiple applications**: Style transfer, object transfiguration, season transfer, photo enhancement

### Advanced Features
- **Mixed precision training**: Faster training with reduced memory usage
- **Distributed training**: Multi-GPU support for faster training
- **TensorBoard logging**: Real-time training monitoring
- **Comprehensive metrics**: FID, Inception Score, and LPIPS evaluation
- **Advanced data augmentation**: MixUp, color jittering, and more
- **Web interface**: Flask-based web application for easy image translation
- **Model evaluation tools**: Comprehensive evaluation scripts
- **Pre-trained models**: Download pre-trained models for quick start
- **Unit tests**: Comprehensive test suite
- **Complete documentation**: API docs and usage guides

## Technologies

- Python
- PyTorch
- TensorFlow (optional)
- CycleGAN
- Image-to-Image Translation
- Cycle Consistency
- Unpaired Training
- Style Transfer
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rskworld/cyclegan-translation.git
cd cyclegan-translation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --dataroot ./datasets/your_dataset --name experiment_name
```

### Testing

```bash
python test.py --dataroot ./datasets/your_dataset --name experiment_name --model test
```

### Advanced Training

```bash
# Mixed precision training
python train_advanced.py --dataroot ./datasets/your_dataset --name experiment_name --mixed_precision

# Distributed training
python train_advanced.py --dataroot ./datasets/your_dataset --name experiment_name --distributed --world_size 4
```

### Evaluation

```bash
python evaluate.py --dataroot ./datasets/your_dataset --name experiment_name
```

### Web Interface

```bash
cd web
python app.py
```

Then open http://localhost:5000 in your browser.

### Jupyter Notebook

Open `CycleGAN_Demo.ipynb` for interactive demonstration and experimentation.

### Download Pre-trained Models

```bash
python scripts/download_pretrained.py horse2zebra
```

## Project Structure

See `PROJECT_STRUCTURE.md` for complete project structure.

Key directories:
- `models/` - Model architectures
- `data/` - Data loading utilities
- `metrics/` - Evaluation metrics (FID, IS, LPIPS)
- `logger/` - Logging utilities (TensorBoard, file logging)
- `tools/` - Advanced tools (augmentation, model utils)
- `web/` - Web interface (Flask application)
- `scripts/` - Utility scripts
- `tests/` - Unit tests
- `docs/` - Documentation

## Dataset Structure

Organize your dataset as follows:
```
datasets/
└── your_dataset/
    ├── trainA/  # Domain A training images
    ├── trainB/  # Domain B training images
    ├── testA/   # Domain A test images
    └── testB/   # Domain B test images
```

## License

This project is provided by RSK World. For more information, visit [rskworld.in](https://rskworld.in)

## Contact

- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

