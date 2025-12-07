# CycleGAN API Documentation

<!--
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
-->

## Models

### CycleGANModel

Main CycleGAN model class.

```python
from models import CycleGANModel

model = CycleGANModel(opt)
```

#### Methods

- `set_input(input)`: Set input data
- `forward()`: Forward pass
- `optimize_parameters()`: Optimize model parameters
- `test()`: Test mode forward pass
- `save_networks(epoch)`: Save model checkpoints
- `load_networks(epoch)`: Load model checkpoints

## Networks

### define_G

Create a generator network.

```python
from models.networks import define_G

netG = define_G(input_nc, output_nc, ngf, netG, norm, use_dropout, init_type, init_gain, gpu_ids)
```

### define_D

Create a discriminator network.

```python
from models.networks import define_D

netD = define_D(input_nc, ndf, netD, n_layers_D, norm, init_type, init_gain, gpu_ids)
```

## Metrics

### MetricsCalculator

Comprehensive metrics calculator.

```python
from metrics import MetricsCalculator

calc = MetricsCalculator(device='cuda')
metrics = calc.calculate_all_metrics(real_images, fake_images)
```

## Logging

### TensorBoardLogger

TensorBoard logging.

```python
from logger import TensorBoardLogger

logger = TensorBoardLogger(log_dir='./logs', experiment_name='exp1')
logger.log_scalar('loss', 0.5, step=100)
logger.log_image('sample', image_tensor, step=100)
```

## Tools

### ModelUtils

Model utility functions.

```python
from tools import ModelUtils

param_count = ModelUtils.count_parameters(model)
ModelUtils.save_checkpoint(model, optimizer, epoch, loss, 'checkpoint.pth')
```

### TrainingUtils

Training utility functions.

```python
from tools import TrainingUtils

lr = TrainingUtils.get_learning_rate(optimizer)
TrainingUtils.set_learning_rate(optimizer, 0.0001)
```

