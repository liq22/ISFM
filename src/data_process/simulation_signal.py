import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

def downsample_signal(signal, factor):
    """
    Downsamples the signal by the given factor.
    
    Args:
        signal (numpy array): The original signal.
        factor (int): The downsampling factor.
        
    Returns:
        numpy array: The downsampled signal.
    """
    return signal[::factor]

class SignalDataset(Dataset):
    def __init__(self, types = 'AM', base_frequencies = 1000, num_samples_per_class=300,
                 sampling_rate=1024, duration=1.0,
                 transform=True, npy_path=None, regenerate=False
                 , downsample_factor=1):
        """
        Args:
            types (list): List of signal types to generate.
            base_frequencies (list): List of base frequencies corresponding to each signal type.
            num_samples_per_class (int): Number of samples to generate per signal type.
            sampling_rate (float): Sampling frequency in Hz.
            duration (float): Duration of the signal in seconds.
            transform (callable, optional): Optional transform to be applied on a sample.
            npy_path (str, optional): Path to the npy file to save/load the dataset.
            regenerate (bool): Whether to regenerate the dataset even if the npy file exists.
        """
        self.types = types
        self.base_frequencies = base_frequencies
        self.num_samples_per_class = num_samples_per_class
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.transform = transform
        self.npy_path = npy_path
        self.regenerate = regenerate
        self.downsample_factor = downsample_factor
        self.snr_db = 0.0

        self.t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        self.signals = []
        self.labels = []

        if npy_path and os.path.exists(npy_path) and not regenerate:
            self.load_dataset(npy_path)
        else:
            self.generate_dataset()
            if npy_path:
                self.save_dataset(npy_path)

    def generate_dataset(self):
        for idx, (signal_type, base_freq) in enumerate(zip(self.types, self.base_frequencies)):
            for _ in range(self.num_samples_per_class):
                signal = self.generate_signal(signal_type, base_freq)
                if self.transform:
                    signal = self.add_noise(signal, self.snr_db)
                self.signals.append(signal)
                self.labels.append(idx)  # Assign an integer label for each type

        # Convert lists to numpy arrays
        self.signals = np.array(self.signals)  # Shape: (2100, L)
        self.labels = np.array(self.labels)    # Shape: (2100,)

    def generate_signal(self, signal_type, base_freq):
        t = self.t
        if signal_type == 'AM':
            # Amplitude Modulated Signal
            carrier = np.cos(2 * np.pi * base_freq * t)
            modulating_freq = base_freq / 10
            modulation_index = 0.5
            modulating_signal = np.cos(2 * np.pi * modulating_freq * t)
            signal = (1 + modulation_index * modulating_signal) * carrier
        elif signal_type == 'FM':
            # Frequency Modulated Signal
            modulating_freq = base_freq / 10
            frequency_deviation = base_freq / 5
            signal = np.cos(2 * np.pi * (base_freq * t + frequency_deviation * np.sin(2 * np.pi * modulating_freq * t)))
        elif signal_type == 'Sine':
            # Basic Sine Wave
            amplitude = 1.0
            phase = 0.0
            signal = amplitude * np.sin(2 * np.pi * base_freq * t + phase)
        elif signal_type == 'MultiHarmonic':
            # Multi-Harmonic Signal
            y = np.zeros_like(t)
            amplitudes = [1.0 / n for n in range(1, 6)]  # Decreasing amplitudes
            phases = [0.0 for _ in range(1, 6)]
            for i, (amplitude, phase) in enumerate(zip(amplitudes, phases), start=1):
                y += amplitude * np.sin(2 * np.pi * i * base_freq * t + phase)
            signal = y
        elif signal_type == 'RandomHarmonic':
            # Random Harmonic Signal
            y = np.zeros_like(t)
            num_components = 5
            amplitudes = np.random.uniform(0.5, 1.0, num_components)
            frequencies = np.random.uniform(base_freq, base_freq * 5, num_components)
            for amplitude, frequency in zip(amplitudes, frequencies):
                y += amplitude * np.sin(2 * np.pi * frequency * t)
            signal = y
        elif signal_type == 'AM_FM':
            # Sum of AM and FM Signals
            am_signal = self.generate_signal('AM', base_freq)
            fm_signal = self.generate_signal('FM', base_freq)
            signal = am_signal + fm_signal
        elif signal_type == 'Complex':
            # Sum of AM_FM and MultiHarmonic Signals
            am_fm_signal = self.generate_signal('AM_FM', base_freq)
            multi_harmonic_signal = self.generate_signal('MultiHarmonic', base_freq)
            signal = am_fm_signal + multi_harmonic_signal
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        return signal
    def add_noise(self, signal, snr_db=0):
        # 计算信号功率
        signal_power = np.mean(signal ** 2)
        # 计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))
        # 生成噪声
        noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
        # 添加噪声到信号
        signal_noisy = signal + noise
        return signal_noisy
        
    def save_dataset(self, npy_path):
        np.savez(npy_path, signals=self.signals, labels=self.labels)

    def load_dataset(self, npy_path):
        data = np.load(npy_path,allow_pickle=True)
        self.signals = data['signals']
        self.labels = data['labels']

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        if self.downsample_factor > 1:
            signal = downsample_signal(signal, self.downsample_factor)
        # if self.transform:
        #     signal = self.transform(signal)
        # Reshape signal to (L, 1)
        signal = np.expand_dims(signal, axis=-1)  # Shape: (L, 1)
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    
def get_dataloaders(dataset, batch_size=32, test_size=0.2, val_size=0.1):
    """
    Splits the dataset into training, validation, and test sets, and returns DataLoaders.

    Args:
        dataset (Dataset): The dataset to split.
        batch_size (int): Batch size for DataLoaders.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split.
    """
    # Calculate sizes
    total_size = len(dataset)
    test_size = int(total_size * test_size)
    val_size = int(total_size * val_size)
    train_size = total_size - test_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Define the dataset
    # Define signal types and base frequencies
    signal_types = ['AM', 'FM', 'Sine', 'MultiHarmonic', 'RandomHarmonic', 'AM_FM', 'Complex']
    base_frequencies = [50, 60, 70, 80, 90, 100, 110]  # Example base frequencies for each signal type

    # Create the dataset
    dataset = SignalDataset(signal_types, base_frequencies, sampling_rate=1024, duration=10, npy_path='data.npz',regenerate=True)


    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=32)

    # 从 dataset 中获取一个样本
    data = dataset.signals

    print(data.shape)