"""Image filtering functions for activation maps.

This module provides filtering techniques, such as the rank-based Gaussian filter,
to denoise and smooth Token Activation Maps for better visualization.
"""

import numpy as np
from typing import Optional, Tuple


def rank_gaussian_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply a rank-based Gaussian-weighted filter for robust activation map denoising.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    if kernel_size <= 1:
        return img.astype(np.float64, copy=True)

    filtered_img = np.zeros_like(img, dtype=np.float64)
    pad_width = kernel_size // 2
    padded_img = np.pad(img, pad_width, mode="reflect")

    ax = np.arange(kernel_size ** 2) - kernel_size ** 2 // 2
    try:
        windows = np.lib.stride_tricks.sliding_window_view(
            padded_img, (kernel_size, kernel_size)
        )
        h, w = windows.shape[:2]
        windows = windows.reshape(h * w, -1)

        sorted_windows = np.sort(windows, axis=1)
        means = sorted_windows.mean(axis=1)
        stds = sorted_windows.std(axis=1)

        valid = means > 0
        sigma = np.zeros_like(means)
        sigma[valid] = stds[valid] / means[valid]

        weights = np.zeros_like(sorted_windows)
        valid_sigma = sigma > 0
        if np.any(valid_sigma):
            ax_sq = (ax ** 2).astype(sorted_windows.dtype)
            denom = 2 * (sigma[valid_sigma] ** 2)[:, None]
            weights[valid_sigma] = np.exp(-ax_sq[None, :] / denom)
            weight_sums = weights[valid_sigma].sum(axis=1, keepdims=True)
            weights[valid_sigma] = np.divide(
                weights[valid_sigma],
                weight_sums,
                out=np.zeros_like(weights[valid_sigma]),
                where=weight_sums != 0
            )

        values = np.zeros_like(means)
        if np.any(valid_sigma):
            values[valid_sigma] = (sorted_windows[valid_sigma] * weights[valid_sigma]).sum(axis=1)

        filtered_img = values.reshape(h, w)
        return filtered_img
    except Exception:
        for i in range(pad_width, img.shape[0] + pad_width):
            for j in range(pad_width, img.shape[1] + pad_width):
                window = padded_img[
                    i - pad_width:i + pad_width + 1,
                    j - pad_width:j + pad_width + 1
                ]

                sorted_window = np.sort(window.flatten())
                mean = sorted_window.mean()

                if mean > 0:
                    sigma = sorted_window.std() / mean
                    kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2))
                    kernel = kernel / np.sum(kernel)
                    value = (sorted_window * kernel).sum()
                else:
                    value = 0.0

                filtered_img[i - pad_width, j - pad_width] = value

        return filtered_img


def apply_filter_to_map(
    flat_map: np.ndarray,
    grid: Optional[Tuple[int, int, int]],
    kernel_size: int
) -> np.ndarray:
    if grid:
        t_size, height, width = grid
        frames = np.array_split(flat_map, t_size)
        filtered_frames = []
        for frame in frames:
            if len(frame) == height * width:
                frame_2d = frame.reshape(height, width)
                filtered_2d = rank_gaussian_filter(frame_2d, kernel_size)
                filtered_frames.append(filtered_2d.flatten())
            else:
                filtered_frames.append(frame)
        return np.concatenate(filtered_frames)

    n_tokens = len(flat_map)
    # If no grid is provided, we have to guess, but this is unreliable for non-square images
    side = int(np.sqrt(n_tokens))
    height = side
    width = n_tokens // side
    
    # Check if this guess actually matches the token count
    if height * width != n_tokens:
        # If it doesn't match, we can't really reshape it reliably without knowing the grid
        return flat_map

    map_2d = flat_map[:height * width].reshape(height, width)
    filtered_2d = rank_gaussian_filter(map_2d, kernel_size)
    if height * width < n_tokens:
        return np.concatenate([filtered_2d.flatten(), flat_map[height * width:]])
    return filtered_2d.flatten()
