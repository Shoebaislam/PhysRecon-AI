"""Single-coil Cartesian MRI reconstruction primitives (NumPy, scikit-image)."""
import numpy as np
from skimage.restoration import denoise_tv_chambolle


def fft2c(x):
    """Centered orthonormal 2D FFT, operating on last two axes."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def ifft2c(k):
    """Centered orthonormal inverse 2D FFT."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), norm='ortho'), axes=(-2, -1))


def cartesian_mask(shape, acceleration=4, acs_lines=0):
    """Uniform phase-encoding undersampling; optional fully sampled center.

    acs_lines=0 gives precisely 4x for 256 rows and acceleration=4.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float64)
    mask[::acceleration, :] = 1.0
    if acs_lines:
        center = h // 2
        mask[max(0, center-acs_lines//2):min(h, center+(acs_lines+1)//2), :] = 1
    return mask


def corrupt_measurements(kspace_sampled, mask, sigma_fraction, rng):
    """Complex Gaussian noise relative to RMS magnitude of measured coefficients.

    E|noise|^2 = sigma_fraction^2 * RMS(measured kspace)^2, on sampled entries.
    """
    if sigma_fraction == 0:
        return kspace_sampled.copy()
    rms = np.sqrt(np.mean(np.abs(kspace_sampled[mask != 0])**2))
    noise = ((rng.standard_normal(mask.shape) + 1j*rng.standard_normal(mask.shape))
             / np.sqrt(2)) * (sigma_fraction * rms)
    return kspace_sampled + mask * noise


def tv_reconstruct(y, mask, lam=0.015, iterations=80, tv_inner=30):
    """FISTA for min_{x real} 0.5||M F x - y||_2^2 + lam TV(x).

    Uses the real-valued constrained-image model. Gradient = Re(F^-1 M (M F x-y)).
    F is orthonormal; Lipschitz constant <= 1, hence step=1.
    denoise_tv_chambolle provides a numerical TV proximal step.
    No projection that overwrites measured coefficients is used: fidelity is SOFT,
    not exact, which is appropriate when acquisitions are noisy.
    """
    x = np.real(ifft2c(y))
    z = x.copy()
    t = 1.0
    for _ in range(iterations):
        residual = mask * fft2c(z) - y
        gradient = np.real(ifft2c(mask * residual))
        next_x = denoise_tv_chambolle(z - gradient, weight=lam,
                                      max_num_iter=tv_inner, eps=1e-5)
        next_t = (1 + np.sqrt(1 + 4*t*t)) / 2
        z = next_x + ((t - 1) / next_t) * (next_x - x)
        x, t = next_x, next_t
    return x


def measured_relative_residual(image, y, mask):
    return float(np.linalg.norm(mask * fft2c(image) - y) / max(np.linalg.norm(y), 1e-12))
