"""OPTIONAL extension: retrospectively undersample an existing fastMRI SINGLECOIL raw k-space slice.

Usage: python fastmri_optional.py path/to/singlecoil_file.h5 --slice 10
Requires separately obtained fastMRI dataset access. This script does NOT download it.
The raw source measurements are genuine MRI k-space; 4x masking is RETROSPECTIVE
and is NOT a prospective 4x acquisition. Reconstruction target is full-data
inverse FFT of the same slice, not an independently validated ground truth.
"""
import argparse
from pathlib import Path
import csv
import time
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from recon import fft2c, ifft2c, cartesian_mask, tv_reconstruct


def main():
    p = argparse.ArgumentParser()
    p.add_argument('file', type=Path)
    p.add_argument('--slice', type=int, default=10)
    p.add_argument('--iterations', type=int, default=80)
    p.add_argument('--lambda-tv', type=float, default=0.015)
    a = p.parse_args()
    with h5py.File(a.file, 'r') as h5:
        k = np.asarray(h5['kspace'][a.slice])
    if k.ndim != 2:
        raise ValueError('Only fastMRI SINGLECOIL data with 2D slices is supported; multicoil needs sensitivity-aware modeling.')
    # fastMRI k-space uses centered k-space storage convention.
    full_image = ifft2c(k)
    # Real-valued TV in this MVP is not a full complex-valued phase-aware MRI model.
    # Estimate global phase from center-weighted phase; data phase may vary spatially.
    # To avoid a misleading inversion, use real-valued magnitude synthetic k-space:
    # this is a PHANTOM-LIKE SIMULATION BASED ON REAL MRI IMAGE, not actual raw kspace.
    # For actual raw kspace, implement phase-aware complex regularization first.
    raise NotImplementedError(
        'Raw fastMRI k-space detected successfully (shape=%s). Complex-valued MRI reconstruction '
        'with coil/phase handling is required for scientifically defensible metrics. '
        'The simulated phantom pipeline remains fully runnable.' % (k.shape,)
    )

if __name__ == '__main__':
    main()
