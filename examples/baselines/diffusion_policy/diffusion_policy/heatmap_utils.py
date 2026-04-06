"""Utilities for generating and processing heatmaps from bounding boxes."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def generate_gaussian_heatmap(
    center_x: float,
    center_y: float,
    height: int,
    width: int,
    sigma: float = 5.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate a Gaussian heatmap given a center point.
    
    Args:
        center_x: X coordinate of center (in pixel coordinates)
        center_y: Y coordinate of center (in pixel coordinates)
        height: Height of heatmap
        width: Width of heatmap
        sigma: Standard deviation of Gaussian kernel
        device: Device to create tensor on
        
    Returns:
        Gaussian heatmap of shape (height, width)
    """
    # Create coordinate grids
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    
    # Compute Gaussian
    kernel = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma ** 2))
    
    # Normalize to [0, 1]
    kernel = kernel / (kernel.max() + 1e-8)
    
    heatmap = torch.from_numpy(kernel).to(dtype=torch.float32)
    if device is not None:
        heatmap = heatmap.to(device=device)
    
    return heatmap


def bbox_centers_to_heatmap_batch(
    init_centers: torch.Tensor,
    goal_centers: torch.Tensor,
    heights: torch.Tensor,
    widths: torch.Tensor,
    heatmap_h: int = 32,
    heatmap_w: int = 32,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Convert normalized bbox centers to heatmap batch.
    
    Args:
        init_centers: (B, 2) normalized init centers in [0, 1]
        goal_centers: (B, 2) normalized goal centers in [0, 1]
        heights: (B,) image heights in pixels
        widths: (B,) image widths in pixels
        heatmap_h: Height of output heatmaps
        heatmap_w: Width of output heatmaps
        sigma: Gaussian sigma in heatmap coordinates
        
    Returns:
        Heatmap tensor (B, 2, heatmap_h, heatmap_w)
        First channel is init, second is goal
    """
    batch_size = init_centers.shape[0]
    device = init_centers.device
    
    heatmaps = []
    
    x_scale = max(heatmap_w - 1, 1)
    y_scale = max(heatmap_h - 1, 1)

    for b in range(batch_size):
        # Convert normalized coords to heatmap pixel coords
        init_x = float(init_centers[b, 0].item()) * x_scale
        init_y = float(init_centers[b, 1].item()) * y_scale
        goal_x = float(goal_centers[b, 0].item()) * x_scale
        goal_y = float(goal_centers[b, 1].item()) * y_scale

        init_x = min(max(init_x, 0.0), float(x_scale))
        init_y = min(max(init_y, 0.0), float(y_scale))
        goal_x = min(max(goal_x, 0.0), float(x_scale))
        goal_y = min(max(goal_y, 0.0), float(y_scale))
        
        # Generate heatmaps
        init_hm = generate_gaussian_heatmap(
            init_x, init_y, heatmap_h, heatmap_w, sigma=sigma, device=device
        )
        goal_hm = generate_gaussian_heatmap(
            goal_x, goal_y, heatmap_h, heatmap_w, sigma=sigma, device=device
        )
        
        heatmaps.append(torch.stack([init_hm, goal_hm], dim=0))
    
    return torch.stack(heatmaps, dim=0)  # (B, 2, H, W)


def bbox_centers_to_heatmap_multi_camera(
    centers_per_camera: list,
    heatmap_h: int = 32,
    heatmap_w: int = 32,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Convert bbox centers from multiple cameras to multi-channel heatmap.
    
    Args:
        centers_per_camera: List of (init_centers, goal_centers) for each camera.
                           Each has shape (B, 2), values in [0, 1]
        heatmap_h: Height of output heatmaps
        heatmap_w: Width of output heatmaps
        sigma: Gaussian sigma in heatmap coordinates
        
    Returns:
        Heatmap tensor (B, num_cameras*2, heatmap_h, heatmap_w)
    """
    num_cameras = len(centers_per_camera)
    batch_size = centers_per_camera[0][0].shape[0]
    device = centers_per_camera[0][0].device
    
    # Initialize batch container: all_heatmaps[b] will contain list of heatmap channels for batch b
    all_heatmaps = [[] for _ in range(batch_size)]
    
    x_scale = max(heatmap_w - 1, 1)
    y_scale = max(heatmap_h - 1, 1)

    for cam_idx in range(num_cameras):
        init_centers, goal_centers = centers_per_camera[cam_idx]
        
        for b in range(batch_size):
            init_x = float(init_centers[b, 0].item()) * x_scale
            init_y = float(init_centers[b, 1].item()) * y_scale
            goal_x = float(goal_centers[b, 0].item()) * x_scale
            goal_y = float(goal_centers[b, 1].item()) * y_scale

            init_x = min(max(init_x, 0.0), float(x_scale))
            init_y = min(max(init_y, 0.0), float(y_scale))
            goal_x = min(max(goal_x, 0.0), float(x_scale))
            goal_y = min(max(goal_y, 0.0), float(y_scale))
            
            init_hm = generate_gaussian_heatmap(
                init_x, init_y, heatmap_h, heatmap_w, sigma=sigma, device=device
            )
            goal_hm = generate_gaussian_heatmap(
                goal_x, goal_y, heatmap_h, heatmap_w, sigma=sigma, device=device
            )
            
            all_heatmaps[b].append(init_hm)
            all_heatmaps[b].append(goal_hm)
    
    # Stack into (B, num_cameras*2, H, W)
    result = []
    for b in range(batch_size):
        result.append(torch.stack(all_heatmaps[b], dim=0))
    
    return torch.stack(result, dim=0)


def normalize_heatmap(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize heatmap to [0, 1] range."""
    hmin = heatmap.amin(dim=(-2, -1), keepdim=True)
    hmax = heatmap.amax(dim=(-2, -1), keepdim=True)
    return (heatmap - hmin) / (hmax - hmin + eps)


def heatmap_to_centers(
    heatmap: torch.Tensor,
    normalized: bool = True,
    heatmap_h: Optional[int] = None,
    heatmap_w: Optional[int] = None,
) -> torch.Tensor:
    """Extract center coordinates from heatmap using maximum activation.
    
    Args:
        heatmap: Tensor of shape (B, C, H, W) or (C, H, W)
        normalized: If True, return normalized coords in [0, 1]; else pixel coords
        heatmap_h: Original image height (for denormalization)
        heatmap_w: Original image width (for denormalization)
        
    Returns:
        Tensor of shape (..., 2) with (x, y) coordinates
    """
    if heatmap.ndim == 3:
        heatmap = heatmap.unsqueeze(0)
    
    batch_size, channels, h, w = heatmap.shape
    
    # Find maximum for each channel
    heatmap_flat = heatmap.reshape(batch_size, channels, -1)
    max_idx = heatmap_flat.argmax(dim=-1)  # (B, C)
    
    max_y = (max_idx // w).float()
    max_x = (max_idx % w).float()
    
    if normalized:
        max_x = max_x / max(w - 1, 1)
        max_y = max_y / max(h - 1, 1)
    
    centers = torch.stack([max_x, max_y], dim=-1)  # (B, C, 2)
    
    return centers
