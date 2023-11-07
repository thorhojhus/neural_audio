# Deep Learning Audio Project for the DTU course 02456 

## Overview
This project aims to finetune the descript-audio-codec for speech enhancement and noise removal. 

## Components

- **Data Preparation**: Custom dataloaders and datasets manage voice and noise data, and creating noisy speech data. 
- **Model Architecture**: Utilizes `dac` (Descript Audio Codec), a high-fidelity general neural audio codec, with both generator and discriminator networks for adversarial training.
- **Loss Functions**: A diverse set of losses, including L1, Mel-Spectrogram, SI-SDR, and GAN losses, are used to capture various aspects of audio reconstruction quality.
- **Training Routine**: A training loop with loss weighting, noise addition, and gradient clipping to create stable learning.
- **Evaluation**: The model's output is periodically saved for quality inspection, and performance metrics are tracked during training.

## Usage

The repository includes scripts used for experimentation. So far it includes the training routine (`training_audiotools.py`) with a range of losses and a simpler SI-SDR-focused training (`simple_training_SI-SDR.py`) (unstable). 

- **training_audiotools.py**: Advanced script utilizing custom audio loaders, various loss functions, and a detailed training loop with audio sample saving.
- **simple_training_SI-SDR.py**: A streamlined training script focusing on optimizing the Scale-Invariant Signal-to-Distortion Ratio metric.

### Training

To start the training process, run the desired script with:

```sh
python3 training_audiotools.py
```
or
```sh
python3 simple_training_SI-SDR.py
```

Adjust hyperparameters such as learning rates, batch sizes, and epochs within the scripts as needed.

## Output

Models, logs, and audio samples are saved in the `./output/` directory, allowing for incremental evaluation of the audio processing quality.

---

Please refer to the individual scripts for detailed parameter settings and training customizations.
