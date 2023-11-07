import dac
import os
import random
import soundfile as sf
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import add_noise
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# root_dir = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/'
# clean_dir = root_dir + 'clean_fullband/vctk_wav48_silence_trimmed/'
# noise_dir = root_dir + 'noise_fullband'

# local paths
out_dir = './output/'
root_dir = './data/'
clean_dir = root_dir + 'voice_fullband/'
noise_dir = root_dir + 'noise_fullband/'

batch_size = 2
target_length = 2**15  # needs to be a power of 2
learning_rate = 5e-5
clip_value = 1.0 
num_epochs = 10 

def audio_collate_fn(batch):
    # Initialize lists to store processed waveforms
    processed_clean_waveforms = []
    processed_noise_waveforms = []

    # Process each sample in the batch
    for clean_waveform, noise_waveform in batch:
        current_length = clean_waveform.size(0)

        # If the waveform is shorter than the target length, we pad it
        if current_length < target_length:
            # Pad the waveforms to the target length
            padded_clean_waveform = torch.nn.functional.pad(clean_waveform, (0, target_length - current_length))
            padded_noise_waveform = torch.nn.functional.pad(noise_waveform, (0, target_length - current_length))
        else:
            # Randomly select a start point for trimming if waveform is longer than target
            start = torch.randint(0, current_length - target_length + 1, (1,)).item()
            end = start + target_length

            # Trim the waveforms to the target length
            padded_clean_waveform = clean_waveform[start:end]
            padded_noise_waveform = noise_waveform[start:end]

        # Add the processed waveforms to the lists
        processed_clean_waveforms.append(padded_clean_waveform)
        processed_noise_waveforms.append(padded_noise_waveform)

    # Stack all the processed waveforms together to create batches
    batched_clean_waveforms = torch.stack(processed_clean_waveforms)
    batched_noise_waveforms = torch.stack(processed_noise_waveforms)

    snr_value = 10  # signal-to-noise ratio in dB
    snr_tensor = torch.tensor([snr_value], dtype=torch.float32)

    # The following line adds noise to the batch of clean waveforms to create noisy waveforms
    noisy_waveforms = add_noise(batched_clean_waveforms, batched_noise_waveforms, snr_tensor).view(-1, 1, target_length)
    clean_waveforms = batched_clean_waveforms.view(-1, 1, target_length)

    return clean_waveforms, noisy_waveforms

class MixedAudioDataset(Dataset):
    def __init__(self, clean_dir, noise_dir):
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.clean_file_list = self._get_file_list(clean_dir)
        self.noise_file_list = self._get_file_list(noise_dir)

    def _get_file_list(self, directory):
        # Walk the directory to get the list of audio files
        file_list = []
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.wav'):
                    file_list.append(os.path.join(subdir, file))
        return file_list

    def __len__(self):
        # The length is determined by the number of clean files
        return len(self.clean_file_list)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the clean and noise audio paths
        clean_audio_path = self.clean_file_list[idx]
        noise_audio_path = random.choice(self.noise_file_list)  # Randomly select a noise file
        
        # Load the clean and noise waveforms
        clean_waveform, _ = sf.read(clean_audio_path, dtype='float32')
        noise_waveform, _ = sf.read(noise_audio_path, dtype='float32')

        clean_waveform = torch.from_numpy(clean_waveform)
        noise_waveform = torch.from_numpy(noise_waveform)

        # Ensure noise is the same length as the clean waveform
        min_len = min(len(clean_waveform), len(noise_waveform))
        clean_waveform = clean_waveform[:min_len]
        noise_waveform = noise_waveform[:min_len]

        return clean_waveform.squeeze(0), noise_waveform.squeeze(0)

# Load model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path).cuda()
model.train()

# Initialize loss function and optimizer
si_sdr_metric = ScaleInvariantSignalDistortionRatio().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize dataset and dataloader
dataset = MixedAudioDataset(clean_dir=clean_dir, noise_dir=noise_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=audio_collate_fn)


for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (clean, noise) in enumerate(dataloader):
        clean, noise = clean.cuda(), noise.cuda()

        optimizer.zero_grad()
        recon = model(noise)["audio"]

        loss = -si_sdr_metric(recon, clean)

        # Check for NaN in loss and skip backprop if detected
        if not torch.isnan(loss):
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()

            running_loss += loss.item()

        # Print statistics
        if (i + 1) % 100 == 0:
            average_loss = running_loss / 100 if running_loss != 0 else running_loss
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {average_loss:.4f}')
            running_loss = 0.0

            # Save model and reconstructions every epoch
            torchaudio.save(out_dir+f'audio_sample_recon_epoch_{epoch}_batch_{i}.wav', recon[0].cpu().detach(), 48000)
            torchaudio.save(out_dir+f'audio_sample_clean_epoch_{epoch}_batch_{i}.wav', clean[0].cpu().detach(), 48000)
            torchaudio.save(out_dir+f'audio_sample_noise_epoch_{epoch}_batch_{i}.wav', noise[0].cpu().detach(), 48000)
            torch.save(model.state_dict(), out_dir+f'dac_model_epoch_{epoch}_batch_{i}.pth')

print('Finished Training')