import argparse
import json
import os
import warnings

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.audio import SignalDistortionRatio as SDR

from audiotools import AudioSignal
from audiotools.data.datasets import AudioDataset, AudioLoader
from dac import DAC
from dac.nn.layers import snake, Snake1d
from dac.nn.loss import *
from flatten_dict import flatten, unflatten
from hifiplusplus_discriminator import *

warnings.filterwarnings("ignore")


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description="Run training with specified configuration")
parser.add_argument("-c", "--config", default="config.json", help="Path to configuration file")
args = parser.parse_args()

with open(args.config, 'r') as file: config = json.load(file)

voice_folder = config["voice_folder"]
noise_folder = config["noise_folder"]
lr = config["learning_rate"]
beta1 = config["beta1"]
beta2 = config["beta2"]
batch_size = config["batch_size"] # 16 for 40GB
n_epochs = config["n_epochs"]
do_print = config["do_print"]
use_wandb = config["use_wandb"]
snr = config["snr"]
use_custom_activation = config["use_custom_activation"]
use_pretrained = config["use_pretrained"]
save_state_dict = config["save_state_dict"]
act_func = nn.SiLU() 
n_samples = config["n_samples"]
sample_rate = config["sample_rate"]
use_mos = config["use_mos"]

### Custom activation functions ###
@torch.jit.script
def sse_activation(x, alpha, beta): return torch.exp(beta * torch.sin(alpha * x).pow(2))

@torch.jit.script
def lpa_activation(x, alpha, beta, n):
    sigmoid = 1 / (1 + torch.exp(-alpha * x))
    polynomial = beta * x.pow(n)
    return sigmoid * polynomial

if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Audio-project",
        config=config,
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

voice_dataset_save = AudioDataset(voice_loader,n_examples=n_samples, sample_rate=sample_rate, duration = 5.0)
noise_dataset_save = AudioDataset(noise_loader, n_examples=n_samples, sample_rate=sample_rate, duration = 5.0)

voice_dataset = AudioDataset(voice_loader,n_examples=n_samples, sample_rate=sample_rate, duration = 0.5)
noise_dataset = AudioDataset(noise_loader, n_examples=n_samples, sample_rate=sample_rate, duration = 0.5)

voice_dataloader = DataLoader(voice_dataset, batch_size=batch_size, shuffle=False, collate_fn=voice_dataset.collate, pin_memory=True)
noise_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=True, collate_fn=noise_dataset.collate, pin_memory=True)


# Models
#############################################
if use_pretrained:
    model_path = dac.utils.download(model_type="44khz")
    generator = dac.DAC.load(model_path).to(device)
else:
    generator = dac.DAC().to(device)

if use_custom_activation:
    change_activation_function(generator)

if use_mos:
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
    subjective_model = SQUIM_SUBJECTIVE.get_model().to(device)
    objective_model = SQUIM_OBJECTIVE.get_model().to(device)

MSD = MultiScaleDiscriminator().to(device)
MPD = MultiPeriodDiscriminator().to(device)    

# Optimizers
#############################################
optimizer_g = optim.AdamW(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_msd = optim.AdamW(MSD.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_mpd = optim.AdamW(MPD.parameters(), lr=lr, betas=(beta1, beta2))


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
    "adv/feat_loss": 5.0,
    "adv/gen_loss": 5.0, 
    "vq/commitment_loss": 0.5, 
    "vq/codebook_loss": 1.0, 
    "waveform/loss": 45.0,
    "stft/loss": 1.0,  
    "sisdr/loss": 1.0,
    "sdr/loss": 1.0,
}

# Helper functions
#############################################
def make_noisy(clean : AudioSignal, noise : AudioSignal): return clean.clone().mix(noise, snr=snr)


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

def pretty_print_output(output : dict):
    pretty_output = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in output.items()}
    pretty_output_str = {k: np.array_str(v, precision=4, suppress_small=True) if isinstance(v, np.ndarray) else v for k, v in pretty_output.items()}
    for key, value in pretty_output_str.items():
        print(f"{key}: {value}")

# Training loop function
#############################################
def train_loop(voice_noisy, voice_clean):

    voice_noisy, voice_clean = prep_batch(voice_noisy), prep_batch(voice_clean)

    generator.train()
    MSD.train()
    MPD.train()

    output = {}
    signal = voice_clean["signal"]

    # Generator Forward Pass
    out = generator(voice_noisy.audio_data, voice_noisy.sample_rate)
    recons = AudioSignal(out["audio"], voice_noisy.sample_rate)
    commitment_loss = out["vq/commitment_loss"]
    codebook_loss = out["vq/codebook_loss"]

    ### Discriminator Losses ###
    ############################
    optimizer_mpd.zero_grad()
    optimizer_msd.zero_grad()
    
    # MPD
    y_d_rs_mpd, y_d_gs_mpd, _, _ = MPD(signal.audio_data, recons.clone().detach().audio_data)
    output["adv/disc_loss_mpd"], _, _ = discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)

    # MSD
    y_d_rs_msd, y_d_gs_msd, _, _ = MSD(signal.audio_data, recons.clone().detach().audio_data)
    output["adv/disc_loss_msd"], _, _ = discriminator_loss(y_d_rs_msd, y_d_gs_msd)

    # Update Discriminators
    output["adv/disc_loss_mpd"].backward()
    optimizer_mpd.step()

    output["adv/disc_loss_msd"].backward()
    optimizer_msd.step()

    ### Generator Losses ###
    ########################
    optimizer_g.zero_grad()

    # Feature and Generator Losses
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = MPD(signal.audio_data, recons.audio_data)
    feat_loss_mpd = feature_loss(fmap_rs, fmap_gs)
    msd_gen_loss, _ = generator_loss(y_d_gs)
    
    y_d_r_msd, y_d_g_msd, fmap_r_msd, fmap_g_msd = MSD(signal.audio_data, recons.audio_data)    
    feat_loss_msd = feature_loss(fmap_r_msd, fmap_g_msd)
    mpd_gen_loss, _ = generator_loss(y_d_g_msd)
    
    output["adv/feat_loss"] = feat_loss_msd + feat_loss_mpd
    output["adv/gen_loss"] = msd_gen_loss + mpd_gen_loss

    # Other Losses
    output["stft/loss"] = stft_loss(recons, signal)
    output["mel/loss"] = mel_loss(recons, signal)
    output["waveform/loss"] = waveform_loss(recons, signal)
    output["sisdr/loss"] = sisdr_loss(recons.audio_data, signal.audio_data)
    output["sdr/loss"] = -sdr_loss(recons.audio_data, signal.audio_data)
    output["vq/commitment_loss"] = commitment_loss
    output["vq/codebook_loss"] = codebook_loss
    output["loss"] = sum([v * output[k] for k, v in loss_weights.items() if k in output])

    # Update Generator
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
    
    if use_mos:
        output["MOS"] = subjective_model(recons.audio_data.squeeze(1), signal.audio_data.squeeze(1)).mean()
        
        stoi, pesq, si_sdr = objective_model(recons.audio_data.squeeze(1))
        output["STOI"],output["PESQ"], output["SI-SDR"] = stoi.mean(), pesq.mean(), si_sdr.mean()

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

    recons_path = f"./output/recons_e{epoch}b{i}.wav"
    noisy_path = f"./output/noisy_e{epoch}b{i}.wav"
    clean_path = f"./output/clean_e{epoch}b{i}.wav"

    recons.write(recons_path)
    noisy.cpu().write(noisy_path)
    clean["signal"].cpu().write(clean_path)
    
    if use_wandb:
        wandb.log({"Reconstructed Audio": wandb.Audio(recons_path, caption=f"Reconstructed Epoch {epoch} Batch {i}"),
                "Noisy Audio": wandb.Audio(noisy_path, caption=f"Noisy Epoch {epoch} Batch {i}"),
                "Clean Audio": wandb.Audio(clean_path, caption=f"Clean Epoch {epoch} Batch {i}")})

# Training loop
#############################################
print("Starting training")
for epoch in range(n_epochs):

    for i, (voice_clean, noise) in enumerate(zip(voice_dataloader, noise_dataloader)):
        
        voice_noisy = make_noisy(voice_clean["signal"], noise["signal"]).to(device)

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
            
            if (i%1000 == 0):
                if save_state_dict:
                    torch.save(generator.state_dict(), f"./models/dac_hifi_e{epoch}_it{i}.pth")