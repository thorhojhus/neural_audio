import torch.optim as optim

import dac
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import numpy as np
import random
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

clean_dir = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/'
noise_dir = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband'
batch_size = 12

class AudioDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the audio files.
        """
        self.root_dir = root_dir
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        # Walk the directory to get the list of audio files
        file_list = []
        for subdir, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_list.append(os.path.join(subdir, file))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.file_list[idx]
        waveform, sample_rate = sf.read(audio_path, dtype='float32')
        
        # Convert to PyTorch tensor
        waveform = torch.from_numpy(waveform)
        if waveform.ndim == 1:
            # If mono, add a dimension to mimic a batch
            waveform = waveform.unsqueeze(0)
        
        return waveform, sample_rate

def audio_collate(batch):
    """
    Collate function for the DataLoader to randomly cut segments of the same length from audio files.
    Args:
        batch (list): A list of tuples (waveform, sample_rate).
    
    Returns:
        torch.Tensor: Batch of audio segments with the same length.
    """
    target_length = 16384 # needs to be base 2
    batch_segments = []
    
    for waveform, sample_rate in batch:
        # Ensure the waveform is long enough to get a segment
        if waveform.size(-1) >= target_length:
            # Randomly choose the start position for slicing
            max_start_idx = waveform.size(-1) - target_length
            start_idx = random.randint(0, max_start_idx)
            segment = waveform[..., start_idx:start_idx+target_length]
        else:
            # If the waveform is shorter than the target length, we pad it
            padding_size = target_length - waveform.size(-1)
            # You can choose different padding strategies here
            segment = torch.nn.functional.pad(waveform, (0, padding_size), 'constant', 0)
        
        batch_segments.append(segment)

    # Stack all the segments along a new dimension
    batch_segments = torch.stack(batch_segments, dim=0)
    
    return batch_segments



audio_dataset = AudioDataset(root_dir=clean_dir)
noise_dataset = AudioDataset(root_dir=noise_dir)

# Now, you can use this collate function in the DataLoader.
clean_audio_dataloader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True, collate_fn=audio_collate)

noise_audio_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False, collate_fn=audio_collate)

model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path).cuda()

si_sdr = ScaleInvariantSignalDistortionRatio().cuda()

# Set up the optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Loss criterion
criterion = torch.nn.MSELoss()

# Set the model to training mode
model.train()

num_epochs = 10  # Define the number of epochs for training

for epoch in range(num_epochs):
    running_loss = 0.0
    i = 0
    for clean_audio, noise_audio in zip(clean_audio_dataloader, noise_audio_dataloader):
        # Ensure the shapes are the same for clean and noise audio
        assert clean_audio.shape == noise_audio.shape
        
        # Move data to the appropriate device (GPU in this case)
        clean_audio = clean_audio.cuda()
        noise_audio = noise_audio.cuda()

        # Calculate the noise to add based on SNR
        snr = torch.Tensor([5]).cuda()
        noise_clean_combined = torchaudio.functional.add_noise(clean_audio.squeeze(1), noise_audio.squeeze(1), snr)
        noise_clean_combined = noise_clean_combined.unsqueeze(1)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass to get the reconstructed audio from the noisy input
        z, codes, latents, _, _ = model.encode(noise_clean_combined, 48000)
        recon_audio = model.decode(z)

        # Calculate the loss (we're using SI-SDR here, but you might need to use a different loss for training)
        loss = -si_sdr(recon_audio, clean_audio)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
        i += 1

    # Print epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(clean_audio_dataloader)}")

    # Validation step could be added here if you have a validation set

print('Finished Training')

# Save trained model
torch.save(model.state_dict(), 'model.pth')
