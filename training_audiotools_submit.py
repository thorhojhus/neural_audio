from audiotools.data.datasets import AudioDataset, AudioLoader
from audiotools import AudioSignal
from flatten_dict import flatten, unflatten

import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np

import dac
from dac.nn.loss import L1Loss, MelSpectrogramLoss, SISDRLoss, MultiScaleSTFTLoss, GANLoss
import wandb



voice_folder = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/'
noise_folder = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband'


lr = 1e-4
batch_size = 20 # with duration 1.0 should fill up 40GB
batch_size = 2
n_epochs = 2 
snr = 5

wandb.init(
    # set the wandb project where this run will be logged
    project="Audio-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "VQ-VAE",
    "dataset": "VCTK",
    "epochs": n_epochs,
    }
)

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count



# Dataloaders and datasets
#############################################
voice_loader = AudioLoader(sources=[voice_folder], shuffle=False)
voice_count = count_files(voice_folder)
noise_loader = AudioLoader(sources=[noise_folder], shuffle=True)
noise_count = count_files(noise_folder)
print(f"Number of voice files {voice_count}")
print(f"Number of noise files {noise_count}")

voice_dataset = AudioDataset(voice_loader,n_examples=voice_count, sample_rate=44100, duration = 1.0)
noise_dataset = AudioDataset(noise_loader, n_examples=noise_count, sample_rate=44100, duration = 1.0)
voice_dataloader = DataLoader(voice_dataset, batch_size=batch_size, shuffle=False, collate_fn=voice_dataset.collate, pin_memory=True)
print("Setup dataloaders")
# Models
#############################################
model_path = dac.utils.download(model_type="44khz")
generator = dac.DAC.load(model_path).cuda()
discriminator = dac.model.Discriminator().cuda()

# Optimizers
#############################################
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)


# Losses
#############################################
gan_loss = GANLoss(discriminator).cuda()
stft_loss = MultiScaleSTFTLoss().cuda()
mel_loss = MelSpectrogramLoss().cuda()
waveform_loss = L1Loss().cuda()
sisdr_loss = SISDRLoss().cuda()


# Weighting for losses
#############################################
loss_weights = {
    "mel/loss": 15.0, 
    "adv/feat_loss": 2.0, 
    "adv/gen_loss": 1.0, 
    "vq/commitment_loss": 0.25, 
    "vq/codebook_loss": 1.0,
    "stft/loss": 1.0,
    "sisdr/loss": 20.0,
    }

# Helper functions
#############################################
def make_noisy(clean, noise, snr=snr):
    return clean["signal"].clone().mix(noise["signal"], snr=snr)


def prep_batch(batch, device="cuda"):
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

# Training loop function mixed precision
#############################################
scaler = GradScaler()


def train_loop_mixed_precision(voice_noisy, voice_clean):
    voice_noisy, voice_clean = prep_batch(voice_noisy), prep_batch(voice_clean)
    generator.train()
    discriminator.train()
    
    output = {}
    signal = voice_clean["signal"]
    
    # Forward pass with mixed precision
    with autocast():
        out = generator(voice_noisy.audio_data, voice_noisy.sample_rate)
        recons = AudioSignal(out["audio"], voice_noisy.sample_rate)
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]
    with autocast():
        output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)
    
    # Discriminator optimization step
    optimizer_d.zero_grad()
    scaler.scale(output["adv/disc_loss"]).backward()
    scaler.unscale_(optimizer_g)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
    scaler.step(optimizer_d)
    scaler.update()


    with autocast():
        output["stft/loss"] = stft_loss(recons, signal)
        output["mel/loss"] = mel_loss(recons, signal)
        output["waveform/loss"] = waveform_loss(recons, signal)
        output["sisdr/loss"] = sisdr_loss(recons, signal)
        output["adv/gen_loss"], output["adv/feat_loss"] = gan_loss.generator_loss(recons, signal)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss
        output["loss"] = sum([v * output[k] for k, v in loss_weights.items() if k in output])

    # Generator optimization step
    optimizer_g.zero_grad()
    scaler.scale(output["loss"]).backward()
    scaler.unscale_(optimizer_g)
    output["other/grad_norm_g"] = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1e3)
    scaler.step(optimizer_g)
    scaler.update()

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()

    return {k: v.item() for k, v in sorted(output.items())}

# Training loop function
#############################################
def train_loop(voice_noisy,
               voice_clean,
               batch_num):
    voice_noisy, voice_clean = prep_batch(voice_noisy), prep_batch(voice_clean)

    generator.train()
    discriminator.train()

    output = {}
    signal = voice_clean["signal"]

    out = generator(voice_noisy.audio_data, voice_noisy.sample_rate)
    recons = AudioSignal(out["audio"], voice_noisy.sample_rate)
    commitment_loss = out["vq/commitment_loss"]
    codebook_loss = out["vq/codebook_loss"]

    optimizer_d.zero_grad()
    output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)
    output["adv/disc_loss"].backward()
    optimizer_d.step()

    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)


    output["stft/loss"] = stft_loss(recons, signal)
    output["mel/loss"] = mel_loss(recons, signal)
    output["waveform/loss"] = waveform_loss(recons, signal)
    output["sisdr/loss"] = sisdr_loss(recons, signal)
    output["adv/gen_loss"], output["adv/feat_loss"] = gan_loss.generator_loss(recons, signal)
    output["vq/commitment_loss"] = commitment_loss
    output["vq/codebook_loss"] = codebook_loss
    output["loss"] = sum([v * output[k] for k, v in loss_weights.items() if k in output])

    optimizer_g.zero_grad()
    output["loss"].backward()
    optimizer_g.step()
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1e3)

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()
    log_data = {k: v.item() if torch.is_tensor(v) else v for k, v in output.items()}
    #log_data["batch_num"] = batch_num
    wandb.log(log_data)


    return {k: v for k, v in sorted(output.items())}

@torch.no_grad()
def save_samples(epoch, i):
    generator.eval()
    noise, clean = noise_dataset[1], voice_dataset[1]
    noisy = make_noisy(clean, noise).cuda()

    out = generator(noisy.audio_data.cuda(), noisy.sample_rate)["audio"]
    recons = AudioSignal(out.detach().cpu(), 44100)

    # Define file paths
    recons_path = f"./output/recons_e{epoch}b{i}.wav"
    noisy_path = f"./output/noisy_e{epoch}b{i}.wav"
    clean_path = f"./output/clean_e{epoch}b{i}.wav"

    # Save audio files
    recons.write(recons_path)
    noisy.cpu().write(noisy_path)
    clean["signal"].cpu().write(clean_path)

    # Log audio files to WandB
    wandb.log({"Reconstructed Audio": wandb.Audio(recons_path, caption=f"Reconstructed Epoch {epoch} Batch {i}"),
               "Noisy Audio": wandb.Audio(noisy_path, caption=f"Noisy Epoch {epoch} Batch {i}"),
               "Clean Audio": wandb.Audio(clean_path, caption=f"Clean Epoch {epoch} Batch {i}")})

# Training loop
#############################################
print("Starting traing")
for epoch in range(n_epochs):
    print(f"Epoch: {epoch}")
    print()
    for i, voice_clean in enumerate(voice_dataloader):
        
        voice_noisy = make_noisy(voice_clean, noise_dataset[i]).cuda()
            
        out = train_loop(voice_noisy, voice_clean, batch_num=i)
        #pretty_print_output(out)
        if i % 100 == 0:
            print(f"Batch {i}: \n{out}")
            save_samples(epoch, i)
    torch.save(generator.state_dict(), f"./output/dac_model_epoch_{epoch}.pth")