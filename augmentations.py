import numpy as np

def add_gaussian_noise(signal, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, signal.shape)
    return signal + noise

def time_shift(signal, shift_max=10):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift, axis=-1)

def scale_amplitude(signal, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return signal * scale

def random_crop(signal, crop_size):
    start = np.random.randint(0, signal.shape[-1] - crop_size)
    return signal[:, start:start+crop_size]

def apply_augmentations(signal, augmentations=[]):
    for aug in augmentations:
        signal = aug(signal)
    return signal