from audiotools.data.datasets import AudioDataset, AudioLoader
from audiotools import AudioSignal
from flatten_dict import flatten, unflatten
from torchmetrics.audio import SignalDistortionRatio as SDR
import torch
from torch import nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast


import numpy as np
import os
import dac
from hifiplusplus import MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, generator_loss, feature_loss
from dac.nn.layers import snake, Snake1d
from dac.nn.loss import L1Loss, MelSpectrogramLoss, SISDRLoss, MultiScaleSTFTLoss, GANLoss
import wandb

import warnings
warnings.filterwarnings("ignore")

device = "cpu"
gpu_ok = False
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # device_cap = torch.cuda.get_device_capability()
    # if device_cap in ((7, 0), (8, 0), (9, 0)):
    #     gpu_ok = True    

voice_folder = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/'
noise_folder = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband'

#voice_folder = './data/voice_fullband'
#noise_folder = './data/noise_fullband'

lr = 1e-4
batch_size = 4
n_epochs = 1
do_print = True
use_wandb = False
snr = 10
use_custom_activation = True
use_pretrained = True
save_state_dict = False
act_func = nn.SiLU()

if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Audio-project",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "architecture": "descript-audio-codec",
        "dataset": "VCTK",
        "epochs": n_epochs,
        "batch_size" : batch_size,
        "SNR" : snr,
        "Custom activation" : use_custom_activation,
        "Activation function" : act_func.__name__,
        "Pretrained" : use_pretrained,

        }
    )

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

def change_activation_function(model):
    for name, module in model.named_children():
        if isinstance(module, Snake1d):
            setattr(model, name, act_func)
        else:
            change_activation_function(module)


# Dataloaders and datasets
#############################################
voice_loader = AudioLoader(sources=[voice_folder], shuffle=False)
noise_loader = AudioLoader(sources=[noise_folder], shuffle=True)

voice_dataset_save = AudioDataset(voice_loader,n_examples=48000, sample_rate=44100, duration = 7.0)
noise_dataset_save = AudioDataset(noise_loader, n_examples=48000, sample_rate=44100, duration = 7.0)

voice_dataset = AudioDataset(voice_loader,n_examples=48000, sample_rate=44100, duration = 0.5)
noise_dataset = AudioDataset(noise_loader, n_examples=48000, sample_rate=44100, duration = 0.5)

voice_dataloader = DataLoader(voice_dataset, batch_size=batch_size, shuffle=False, collate_fn=voice_dataset.collate, pin_memory=True)
noise_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=True, collate_fn=noise_dataset.collate, pin_memory=True)


# Models
#############################################
if use_pretrained:
    model_path = dac.utils.download(model_type="44khz")
    generator = dac.DAC.load(model_path).to(device)
    print("Pretrained model loaded")
else:
    generator = dac.DAC().to(device)
    print("New model created")

if use_custom_activation:
    change_activation_function(generator)
    print("Custom activation function applied")

if gpu_ok:
    generator = torch.compile(generator, mode="default")
    #discriminator = torch.compile(discriminator, mode="default")
    print("Model compiled for GPU")

msd = MultiScaleDiscriminator().to(device)
mpd = MultiPeriodDiscriminator().to(device)    

#subjective_model = SQUIM_SUBJECTIVE.get_model()

# Optimizers
#############################################
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_msd = optim.Adam(msd.parameters(), lr=lr)
optimizer_mpd = optim.Adam(mpd.parameters(), lr=lr)


# Losses
#############################################

stft_loss = MultiScaleSTFTLoss().to(device)
mel_loss = MelSpectrogramLoss().to(device)
waveform_loss = L1Loss().to(device)
sisdr_loss = SISDRLoss().to(device)
sdr_loss = SDR().to(device)


# Weighting for losses
#############################################
loss_weights = {
    "mel/loss": 45.0,  
    "adv/disc_loss_mpd": 1.0, 
    "adv/disc_loss_msd": 1.0,  
    "adv/gen_loss": 5.0, 
    "vq/commitment_loss": 0.5, 
    "vq/codebook_loss": 2.0, 
    "stft/loss": 1.0,  
    "sisdr/loss": 10.0 
}

if use_wandb:
    wandb.log(loss_weights)

# Helper functions
#############################################
def make_noisy(clean, noise):
    return clean["signal"].clone().mix(noise["signal"], snr=snr)


def prep_batch(batch):
    if isinstance(batch, dict):
        batch = flatten(batch)
        for key, val in batch.items():
            try:
                batch[key] = val.to(device)
            except:
                pass
        batch = unflatten(batch)
    elif torch.is_tensor(batch):
        batch = batch.to(device)
    elif isinstance(batch, list):
        for i in range(len(batch)):
            try:
                batch[i] = batch[i].to(device)
            except:
                pass
    return batch

def pretty_print_output(output):
    # Detaching and converting tensors to numpy arrays, if applicable
    pretty_output = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in output.items()}

    # Formatting numpy arrays to strings with specified precision
    pretty_output_str = {k: np.array_str(v, precision=4, suppress_small=True) if isinstance(v, np.ndarray) else v for k, v in pretty_output.items()}

    # Printing each key-value pair in a formatted manner
    for key, value in pretty_output_str.items():
        print(f"{key}: {value}")

# Training loop function
#############################################
def train_loop(voice_noisy, voice_clean):
    voice_noisy, voice_clean = prep_batch(voice_noisy), prep_batch(voice_clean)

    generator.train()
    msd.train()
    mpd.train()

    output = {}
    signal = voice_clean["signal"]

    # Generator Forward Pass
    out = generator(voice_noisy.audio_data, voice_noisy.sample_rate)
    recons = AudioSignal(out["audio"], voice_noisy.sample_rate)
    commitment_loss = out["vq/commitment_loss"]
    codebook_loss = out["vq/codebook_loss"]

    # Discriminator (MPD and MSD) Forward Pass and Loss Calculation
    y_mpd_hat_r, y_mpd_hat_g, _, _ = mpd(signal, recons.detach())
    loss_disc_mpd, _, _ = discriminator_loss(y_mpd_hat_r, y_mpd_hat_g)
    output["adv/disc_loss_mpd"] = loss_disc_mpd

    y_msd_hat_r, y_msd_hat_g, _, _ = msd(signal, recons.detach())
    loss_disc_msd, _, _ = discriminator_loss(y_msd_hat_r, y_msd_hat_g)
    output["adv/disc_loss_msd"] = loss_disc_msd

    loss_disc_all = loss_disc_mpd + loss_disc_msd
    output["adv/disc_loss_all"] = loss_disc_all

    # Update Discriminators
    optimizer_mpd.zero_grad()
    optimizer_msd.zero_grad()
    loss_disc_all.backward()
    optimizer_mpd.step()
    optimizer_msd.step()

    # Other Losses
    output["stft/loss"] = stft_loss(recons, signal)
    output["mel/loss"] = mel_loss(recons, signal)
    output["waveform/loss"] = waveform_loss(recons, signal)
    output["sisdr/loss"] = sisdr_loss(recons.audio_data, signal.audio_data)
    output["vq/commitment_loss"] = commitment_loss
    output["vq/codebook_loss"] = codebook_loss
    output["loss"] = sum([v * output[k] for k, v in loss_weights.items() if k in output])

    # Update Generator
    optimizer_g.zero_grad()
    output["loss"].backward()
    optimizer_g.step()
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1e3)

    # Logging
    log_data = {k: v.item() if torch.is_tensor(v) else v for k, v in output.items()}
    
    if use_wandb:
        wandb.log(log_data)

    return output


@torch.no_grad()
def val_loop(voice_noisy,
             voice_clean):
    
    voice_noisy, voice_clean = prep_batch(voice_noisy), prep_batch(voice_clean)
    output = {}
    signal = voice_clean["signal"]
    out = generator(voice_noisy.audio_data, voice_noisy.sample_rate)
    recons = AudioSignal(out["audio"], voice_noisy.sample_rate)
    output["SI-SDR"] = sisdr_loss(recons.audio_data, signal.audio_data)
    #output["mos"] = subjective_model(recons.audio_data, signal.audio_data)
    log_data = {k: v.item() if torch.is_tensor(v) else v for k, v in output.items()}
    if use_wandb:
        wandb.log(log_data)
    return {k: v.item() if torch.is_tensor(v) else v for k, v in sorted(output.items())}

@torch.no_grad()
def save_samples(epoch, i):
    noise, clean = noise_dataset_save[i], voice_dataset_save[i]
    noisy = make_noisy(clean, noise).to(device)

    out = generator(noisy.audio_data.to(device), noisy.sample_rate)["audio"]
    recons = AudioSignal(out.detach().cpu(), 44100)

    # Define file paths
    recons_path = f"./output/recons_e{epoch}b{i}.wav"
    noisy_path = f"./output/noisy_e{epoch}b{i}.wav"
    clean_path = f"./output/clean_e{epoch}b{i}.wav"

    # Save audio files
    recons.write(recons_path)
    noisy.cpu().write(noisy_path)
    clean["signal"].cpu().write(clean_path)
    if use_wandb:
        # Log audio files to WandB
        wandb.log({"Reconstructed Audio": wandb.Audio(recons_path, caption=f"Reconstructed Epoch {epoch} Batch {i}"),
                "Noisy Audio": wandb.Audio(noisy_path, caption=f"Noisy Epoch {epoch} Batch {i}"),
                "Clean Audio": wandb.Audio(clean_path, caption=f"Clean Epoch {epoch} Batch {i}")})

# Training loop
#############################################
print("Starting training")

for i, (voice_clean, noise) in enumerate(zip(voice_dataloader, noise_dataloader)):
    
    # Choose one noise file from 
    voice_noisy = make_noisy(voice_clean, noise).to(device)

    generator.train()
    discriminator.train()
    out = train_loop(voice_noisy, voice_clean)

    if (i % 100 == 0) & (i != 0):
        if do_print:
            print(f"\nBatch {i}:\n")
            pretty_print_output(out)
        generator.eval()
        save_samples(0, i)
        output = val_loop(voice_noisy, voice_clean)
        if do_print:
            print("\nValidation:\n")
            pretty_print_output(output)

if save_state_dict:
    torch.save(generator.state_dict(), f"./output/dac_model_{i}.pth")
    torch.save(discriminator.state_dict(), f"./output/discriminator_{i}.pth")