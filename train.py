import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics.audio import SignalDistortionRatio as SDR
from torch.profiler import profile, record_function, ProfilerActivity
import wandb
from audiotools import AudioSignal
from audiotools.data.datasets import AudioDataset, AudioLoader
import dac
from dac.nn.layers import snake, Snake1d
from dac.nn.loss import *
from flatten_dict import flatten, unflatten

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description="Run training with specified configuration")
parser.add_argument("-c", "--config", default="config.json", help="Path to configuration file")
args = parser.parse_args()

with open(args.config, 'r') as file: config = json.load(file)

### Custom activation functions ###
@torch.jit.script
def sse_activation(x, alpha, beta): return torch.exp(beta * torch.sin(alpha * x).pow(2))

@torch.jit.script
def lpa_activation(x, alpha, beta, n): return (1 / (1 + torch.exp(-alpha * x))) * beta * x.pow(n)

custom_act_func = nn.SiLU() 

if config["use_wandb"]:
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
            setattr(model, name, custom_act_func)
        else:
            change_activation_function(module)


# Dataloaders and datasets
#############################################
voice_loader = AudioLoader(sources=[config["voice_folder"]], shuffle=False)
noise_loader = AudioLoader(sources=[config["noise_folder"]], shuffle=True)

voice_dataset_save = AudioDataset(voice_loader,n_examples=config["n_samples"], sample_rate=config["sample_rate"], duration = 5.0)
noise_dataset_save = AudioDataset(noise_loader, n_examples=config["n_samples"], sample_rate=config["sample_rate"], duration = 5.0)

voice_dataset = AudioDataset(voice_loader,n_examples=config["n_samples"], sample_rate=config["sample_rate"], duration = 0.5)
noise_dataset = AudioDataset(noise_loader, n_examples=config["n_samples"], sample_rate=config["sample_rate"], duration = 0.5)

voice_dataloader = DataLoader(voice_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=voice_dataset.collate, pin_memory=True)
noise_dataloader = DataLoader(noise_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=noise_dataset.collate, pin_memory=True)


# Models
#############################################
if config["use_pretrained"]:
    model_path = dac.utils.download(model_type="44khz")
    generator = dac.DAC.load(model_path).to(device)
else:
    generator = dac.DAC().to(device)

if config["use_custom_activation"]:
    change_activation_function(generator)

if config["use_mos"]:
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
    subjective_model = SQUIM_SUBJECTIVE.get_model().to(device)
    objective_model = SQUIM_OBJECTIVE.get_model().to(device)

if config["hpp_disc"]:
    from hifiplusplus_discriminator import *
    MSD = MultiScaleDiscriminator().to(device)
    MPD = MultiPeriodDiscriminator().to(device)

if config["dac_disc"]:
    dac_disc = dac.model.Discriminator().to(device)

# Optimizers
#############################################
optimizer_gen = optim.AdamW(generator.parameters(), lr=config["learning_rate"], betas=(config["beta1"], config["beta2"]))

if config["hpp_disc"]:
    optimizer_msd = optim.AdamW(MSD.parameters(), lr=config["learning_rate"], betas=(config["beta1"], config["beta2"]))
    optimizer_mpd = optim.AdamW(MPD.parameters(), lr=config["learning_rate"], betas=(config["beta1"], config["beta2"]))

if config["dac_disc"]:
    optimizer_dac_disc = optim.AdamW(dac_disc.parameters(), lr=config["learning_rate"], betas=(config["beta1"], config["beta2"]))

scheduler = StepLR(optimizer_gen, step_size=30, gamma=0.1) if config["use_scheduler"] else None

# Losses
#############################################
stft_loss = MultiScaleSTFTLoss().to(device)
mel_loss = MelSpectrogramLoss().to(device)
waveform_loss = L1Loss().to(device)
sisdr_loss = SISDRLoss().to(device)
sdr_loss = SDR().to(device)
gan_loss = GANLoss(dac_disc).to(device) if config["dac_disc"] else None


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
def add_noise(clean : AudioSignal, noise : AudioSignal): return clean.clone().mix(noise, snr=config["snr"])

def pretty_print_output(output : dict):
    pretty_output = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in output.items()}
    pretty_output_str = {k: np.array_str(v, precision=4, suppress_small=True) if isinstance(v, np.ndarray) else v for k, v in pretty_output.items()}
    for key, value in pretty_output_str.items():
        print(f"{key}: {value}")

# Training loop function
#############################################
def train_loop(noisy_signal : AudioSignal, signal : AudioSignal):

    # Set models to train mode
    generator.train()

    if config["hpp_disc"]:
        MSD.train()
        MPD.train()

    if config["dac_disc"]:
        dac_disc.train()
    
    output = {}

    # Generator Forward Pass
    out = generator(noisy_signal.audio_data, noisy_signal.sample_rate)
    recons = AudioSignal(out["audio"], noisy_signal.sample_rate)
    commitment_loss = out["vq/commitment_loss"]
    codebook_loss = out["vq/codebook_loss"]

    # Hif-plus-plus discriminators
    if config["hpp_disc"]:

        # Multi Period Discriminator
        real_mpd, gen_mpd, _, _ = MPD(signal.audio_data, recons.clone().detach().audio_data)
        output["adv/disc_loss_mpd"], _, _ = discriminator_loss(real_mpd, gen_mpd)

        # Multi Scale Discriminator
        real_msd, gen_msd, _, _ = MSD(signal.audio_data, recons.clone().detach().audio_data)
        output["adv/disc_loss_msd"], _, _ = discriminator_loss(real_msd, gen_msd)

        # Update Discriminators
        output["adv/disc_loss_mpd"].backward()
        optimizer_mpd.step()
        optimizer_mpd.zero_grad()

        output["adv/disc_loss_msd"].backward()
        optimizer_msd.step()
        optimizer_msd.zero_grad()

        # Feature and Generator Losses
        _, gen_mpd, feat_real_mpd, feat_gen_mpd = MPD(signal.audio_data, recons.audio_data)
        feat_loss_mpd = feature_loss(feat_real_mpd, feat_gen_mpd)
        mpd_gen_loss, _ = generator_loss(gen_mpd)
        
        _, gen_msd, feat_real_msd, feat_gen_msd = MSD(signal.audio_data, recons.audio_data)    
        feat_loss_msd = feature_loss(feat_real_msd, feat_gen_msd)
        msd_gen_loss, _ = generator_loss(gen_msd)
        
        output["adv/feat_loss"] = feat_loss_msd + feat_loss_mpd
        output["adv/gen_loss"] = msd_gen_loss + mpd_gen_loss
    

    # Descript audio codec discriminator
    if config["dac_disc"]:
        output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)
        output["adv/disc_loss"].backward()
        torch.nn.utils.clip_grad_norm_(dac_disc.parameters(), 1e3)
        optimizer_dac_disc.step()
        optimizer_dac_disc.zero_grad()
        
        # Generator Losses
        output["adv/gen_loss"], output["adv/feat_loss"] = gan_loss.generator_loss(recons, signal)

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
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1e3)
    optimizer_gen.step()
    scheduler.step() if config["use_scheduler"] else None
    optimizer_gen.zero_grad()

    # Logging
    log_data = {k: v.item() if torch.is_tensor(v) else v for k, v in output.items()}
    
    if config["use_wandb"]:
        wandb.log(log_data)

    return output


@torch.no_grad()
def val_loop(noisy_signal : AudioSignal,
             signal : AudioSignal):
    generator.eval()

    # Create samples
    print("\nValidation:\n") if config["do_print"] else None
    output = {}
    out = generator(noisy_signal.audio_data, noisy_signal.sample_rate)
    recons = AudioSignal(out["audio"], noisy_signal.sample_rate)

    # Get perceptual metrics
    if config["use_mos"]:
        output["MOS"] = subjective_model(recons.audio_data.squeeze(1), signal.audio_data.squeeze(1)).mean()
        stoi, pesq, si_sdr = objective_model(recons.audio_data.squeeze(1))
        output["STOI"],output["PESQ"], output["SI-SDR"] = stoi.mean(), pesq.mean(), si_sdr.mean()
    
    # Log and print
    log_data = {k: v.item() if torch.is_tensor(v) else v for k, v in output.items()}
    wandb.log(log_data) if config["use_wandb"] else None
    pretty_print_output(log_data) if config["do_print"] else None

@torch.no_grad()
def save_samples(epoch : int, i : int):
    generator.eval()
    # Create samples
    noise, clean = noise_dataset_save[i]["signal"].to(device), voice_dataset_save[i]["signal"].to(device)
    noisy_signal = add_noise(clean, noise)
    out = generator(noisy_signal.audio_data.to(device), noisy_signal.sample_rate)["audio"]
    recons = AudioSignal(out, 44100)

    # Define paths
    recons_path = f"./output/recons_e{epoch}b{i}.wav"
    noisy_path = f"./output/noisy_e{epoch}b{i}.wav"
    clean_path = f"./output/clean_e{epoch}b{i}.wav"

    # Write to disk
    recons.cpu().write(recons_path)
    noisy_signal.cpu().write(noisy_path)
    clean.cpu().write(clean_path)

    print(f"Saved samples to {recons_path}, {noisy_path} and {clean_path}") if config["do_print"] else None
    
    wandb.log({"Reconstructed Audio": wandb.Audio(recons_path, caption=f"Reconstructed Epoch {epoch} Batch {i}"),
               "Noisy Audio": wandb.Audio(noisy_path, caption=f"Noisy Epoch {epoch} Batch {i}"),
               "Clean Audio": wandb.Audio(clean_path, caption=f"Clean Epoch {epoch} Batch {i}")}) if config["use_wandb"] else None

# Training loop
#############################################
print("Starting training") if config["do_print"] else None
for epoch in range(config["n_epochs"]):

    for i, (signal, noise) in enumerate(zip(voice_dataloader, noise_dataloader)):
        signal, noise = signal["signal"].to(device), noise["signal"].to(device)
        noisy_signal = add_noise(signal, noise)

        out = train_loop(noisy_signal, signal)
        if (i%config["val_interval"] == 0) & (i != 0):
            if config["do_print"]:
                print(f"\nBatch {i}:\n")
                pretty_print_output(out)

            save_samples(epoch, i)
            val_loop(noisy_signal, signal)
        if (i%config["save_state_dict_interval"] == 0) & (i != 0):
            if config["save_state_dict"]:
                generator.eval()
                torch.save(generator.state_dict(), f"./models/dac_hifi_e{epoch}_it{i}.pth")