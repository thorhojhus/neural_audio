from audiotools.data.datasets import AudioDataset, AudioLoader
from audiotools import AudioSignal
from flatten_dict import flatten, unflatten

import torch
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import dac
from dac.nn.loss import L1Loss, MelSpectrogramLoss, SISDRLoss, MultiScaleSTFTLoss, GANLoss


voice_folder = "./data/voice_fullband"
noise_folder = "./data/noise_fullband"


lr = 1e-4
batch_size = 4
n_epochs = 10


# Dataloaders and datasets
#############################################
voice_loader = AudioLoader(sources=[voice_folder], shuffle=False)
noise_loader = AudioLoader(sources=[noise_folder], shuffle=True)
voice_dataset = AudioDataset(voice_loader, sample_rate=44100)
noise_dataset = AudioDataset(noise_loader, sample_rate=44100)
voice_dataloader = DataLoader(voice_dataset, batch_size=batch_size, shuffle=False, collate_fn=voice_dataset.collate, pin_memory=True)

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
def make_noisy(clean, noise, snr=10):
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


# Training loop function
#############################################
scaler = GradScaler()


def train_loop(voice_noisy, voice_clean):
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

        output["adv/disc_loss"] = gan_loss.discriminator_loss(recons, signal)

        output["stft/loss"] = stft_loss(recons, signal)
        output["mel/loss"] = mel_loss(recons, signal)
        output["waveform/loss"] = waveform_loss(recons, signal)
        output["sisdr/loss"] = sisdr_loss(recons, signal)
        output["adv/gen_loss"], output["adv/feat_loss"] = gan_loss.generator_loss(recons, signal)
        output["vq/commitment_loss"] = commitment_loss
        output["vq/codebook_loss"] = codebook_loss
        output["loss"] = sum([v * output[k] for k, v in loss_weights.items() if k in output])

    # Discriminator optimization step
    optimizer_d.zero_grad()
    scaler.scale(output["adv/disc_loss"]).backward()
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
    scaler.step(optimizer_d)
    scaler.update()

    # Generator optimization step
    optimizer_g.zero_grad()
    scaler.scale(output["loss"]).backward()
    output["other/grad_norm_g"] = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1e3)
    scaler.step(optimizer_g)
    scaler.update()

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()

    return {k: v.item() for k, v in sorted(output.items())}


# Save audio samples
#############################################
@torch.no_grad()
def save_samples(epoch, i):
    generator.eval()
    noise, clean = noise_dataset[1], voice_dataset[1]
    noisy = make_noisy(clean, noise).cuda()

    out = generator(noisy.audio_data.cuda(), noisy.sample_rate)["audio"]
    recons = AudioSignal(out.detach().cpu(), 44100)

    recons.write(f"./output/recons_e{epoch}b{i}.wav")
    noisy.cpu().write(f"./output/noisy_e{epoch}b{i}.wav")
    clean["signal"].cpu().write(f"./output/clean_e{epoch}b{i}.wav")
    generator.train()



# Training loop
#############################################
for epoch in range(n_epochs):
    for i, voice_clean in enumerate(voice_dataloader):
        
        voice_noisy = make_noisy(voice_clean, noise_dataset[i]).cuda()
        
        #noisy_batch, voice_batch = prep_batch(voice_noisy), prep_batch(voice_clean)
        
        out = train_loop(voice_noisy, voice_clean)
        if i % 100 == 0:
            print(f"Batch {i}: {out}")
            save_samples(epoch, i)
    torch.save(generator.state_dict(), f"./output/dac_model_epoch_{epoch}.pth")