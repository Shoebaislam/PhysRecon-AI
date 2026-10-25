"""PhysRecon-AI: reproducible SIMULATED k-space MRI inverse-reconstruction study."""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

from recon import (fft2c, ifft2c, cartesian_mask, corrupt_measurements,
                   tv_reconstruct, measured_relative_residual)


def get_reference(size):
    return resize(shepp_logan_phantom(), (size, size), mode='reflect',
                  anti_aliasing=True, preserve_range=True).astype(np.float64)


def evaluate(reference, image):
    # Both reconstructions are represented by real intensity arrays, without clipping.
    return dict(psnr_db=float(peak_signal_noise_ratio(reference, image, data_range=1.0)),
                ssim=float(structural_similarity(reference, image, data_range=1.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--lambda-tv', type=float, default=0.015)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    if args.size < 32 or args.size % 4:
        parser.error('--size must be >= 32 and divisible by 4 for exactly 4x')
    if args.iterations < 1 or args.lambda_tv < 0:
        parser.error('iterations must be positive; lambda-tv cannot be negative')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    x = get_reference(args.size)
    mask = cartesian_mask(x.shape, acceleration=4)
    full_kspace = fft2c(x)
    y_clean = mask * full_kspace
    acceleration = mask.size / np.count_nonzero(mask)
    rows, images = [], {}
    for noise in (0.0, 0.01, 0.03):
        y = corrupt_measurements(y_clean, mask, noise, rng)
        start = time.perf_counter()
        zf = np.real(ifft2c(y))
        zf_time = time.perf_counter() - start
        start = time.perf_counter()
        tv = tv_reconstruct(y, mask, lam=args.lambda_tv, iterations=args.iterations)
        tv_time = time.perf_counter() - start
        for method, image, duration in [('zero_filled', zf, zf_time),
                                         ('tv_fista', tv, tv_time)]:
            metrics = evaluate(x, image)
            row = dict(dataset='simulated_shepp_logan', acquisition='retrospective_simulation',
                       method=method, noise_sigma_fraction=noise,
                       acceleration=acceleration, iterations=(args.iterations if method == 'tv_fista' else 0),
                       lambda_tv=(args.lambda_tv if method == 'tv_fista' else 0),
                       runtime_seconds=duration,
                       kspace_relative_residual=measured_relative_residual(image, y, mask), **metrics)
            rows.append(row)
            print(f"noise={noise:.2f}  {method:12}  PSNR={metrics['psnr_db']:.3f} dB"
                  f"  SSIM={metrics['ssim']:.4f}  time={duration:.3f}s"
                  f"  relDC={row['kspace_relative_residual']:.5f}")
        images[noise] = (zf, tv)

    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / 'metrics.json').open('w', encoding='utf-8') as f:
        json.dump(dict(description='Simulated phantom MRI inverse problem; NOT scanner data',
                       transform='centered 2D orthonormal FFT',
                       mask='uniform every 4th ky row; exact 4x without ACS',
                       noise='complex Gaussian noise on acquired coefficients; sigma fraction of measured RMS',
                       rng_seed=args.seed, image_size=args.size,
                       regularization='real-valued TV proximal-gradient FISTA; soft data fidelity',
                       rows=rows), f, indent=2)

    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for i, noise in enumerate((0.0, 0.01, 0.03)):
        imgs = [x, np.log1p(np.abs(y_clean)), *images[noise]]
        labels = ['Reference phantom', 'Simulated sampled k-space',
                  'Zero-filled (real)', 'TV-FISTA (real)']
        for j, (arr, label) in enumerate(zip(imgs, labels)):
            axes[i, j].imshow(arr, cmap='gray', vmin=0 if j != 1 else None,
                              vmax=1 if j != 1 else None)
            axes[i, j].set_title(f'{label}\nnoise={noise:.0%}')
            axes[i, j].axis('off')
    fig.tight_layout()
    fig.savefig(out / 'reconstruction_comparison.png', dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    noises = [0.0, 0.01, 0.03]
    for method in ('zero_filled', 'tv_fista'):
        selection = [r for n in noises for r in rows if r['noise_sigma_fraction'] == n
                     and r['method'] == method]
        axs[0].plot(noises, [r['psnr_db'] for r in selection], marker='o', label=method)
        axs[1].plot(noises, [r['ssim'] for r in selection], marker='o', label=method)
    axs[0].set(ylabel='PSNR (dB)', xlabel='Noise sigma / measured RMS')
    axs[1].set(ylabel='SSIM', xlabel='Noise sigma / measured RMS')
    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / 'noise_robustness.png', dpi=160)
    plt.close(fig)
    print(f'\nSaved actual computed results to: {out.resolve()}')
    print(f'Actual acceleration: {acceleration:.1f}x')


if __name__ == '__main__':
    main()
