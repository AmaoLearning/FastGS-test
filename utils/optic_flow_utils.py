"""
Optical Flow Preprocessing Utilities for 3D Gaussian Splatting.

This module handles:
1. Optical flow estimation using RAFT/GMA models
2. Bidirectional flow computation (forward and backward)
3. Occlusion mask generation via forward-backward consistency check
4. Flow and mask I/O operations

Author: 3DGS Deformable Extension
Date: 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import cv2
from torch.utils.model_zoo import load_url


def forward_backward_consistency_check(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    alpha1: float = 0.005,  # [严格化] 从 0.01 降至 0.005
    alpha2: float = 0.2,    # [严格化] 从 0.5 像素降至 0.2 亚像素
) -> torch.Tensor:
    """
    改进版 Forward-Backward Consistency Check
    """
    squeeze = False
    if flow_fwd.dim() == 3:
        flow_fwd = flow_fwd.unsqueeze(0)
        flow_bwd = flow_bwd.unsqueeze(0)
        squeeze = True

    B, _, H, W = flow_fwd.shape

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=flow_fwd.device),
        torch.arange(W, dtype=torch.float32, device=flow_fwd.device),
        indexing='ij',
    )
    
    # x' = x + F_fwd(x)
    sample_x = grid_x.unsqueeze(0) + flow_fwd[:, 0]  # [B, H, W]
    sample_y = grid_y.unsqueeze(0) + flow_fwd[:, 1]  # [B, H, W]

    # [新增] 严格越界检查 (Out-of-bounds Mask)
    # 任何在前向光流中飞出画面的像素，绝对不可信
    out_of_bounds = (sample_x < 0) | (sample_x > W - 1) | \
                    (sample_y < 0) | (sample_y > H - 1)  # [B, H, W]

    sample_x_norm = 2.0 * sample_x / (W - 1) - 1.0
    sample_y_norm = 2.0 * sample_y / (H - 1) - 1.0
    grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)

    flow_bwd_warped = F.grid_sample(
        flow_bwd, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
    )

    cycle_error = torch.norm(flow_fwd + flow_bwd_warped, dim=1, keepdim=True)
    mag_fwd = torch.norm(flow_fwd, dim=1, keepdim=True)
    mag_bwd_w = torch.norm(flow_bwd_warped, dim=1, keepdim=True)
    
    threshold = alpha1 * (mag_fwd + mag_bwd_w) + alpha2

    valid_mask = (cycle_error < threshold).float()  # [B, 1, H, W]

    # [新增] 将越界像素强制设为 0 (不可信)
    valid_mask = valid_mask.masked_fill(out_of_bounds.unsqueeze(1), 0.0)

    if squeeze:
        valid_mask = valid_mask.squeeze(0)

    return valid_mask


class OpticalFlowProcessor:
    """Process sequences to generate pseudo-ground truth optical flow and occlusion masks."""
    
    def __init__(self, 
                 model_name: str = "raft",
                 device: torch.device = None,
                 small: bool = False):
        """
        Initialize the optical flow processor.
        
        Args:
            model_name: "raft" or "gma" for the flow model
            device: torch device (default: cuda if available)
            small: Use small RAFT model for faster inference
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.small = small
        self.model = self._load_model()
        
    def _load_model(self):
        """Load pretrained RAFT or GMA optical flow model."""
        if self.model_name.lower() == "raft":
            return self._load_raft()
        elif self.model_name.lower() == "gma":
            return self._load_gma()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _load_raft(self):
        """Load RAFT optical flow model."""
        try:
            from raft import RAFT
        except ImportError:
            raise ImportError("RAFT not installed. Install via: pip install git+https://github.com/princeton-vl/RAFT.git")
        
        # RAFT weight loading
        model = RAFT(small=self.small)
        
        # Option 1: Load from torchvision (if available)
        try:
            model_url = 'https://dl.fbaipublicfiles.com/raft/raft-things.pth'
            state_dict = load_url(model_url, progress=True, map_location='cpu')
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: Could not load RAFT pretrained weights: {e}")
            print("Proceeding with random initialization. For best results, download weights manually.")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    @torch.no_grad()
    def estimate_flow(self, 
                     image1: torch.Tensor, 
                     image2: torch.Tensor,
                     num_iters: int = 20) -> torch.Tensor:
        """
        Estimate optical flow from image1 to image2.
        
        Args:
            image1: Source image [B, C, H, W] in range [0, 1] or [0, 255]
            image2: Target image [B, C, H, W] in same range
            num_iters: Number of refinement iterations (RAFT specific)
            
        Returns:
            flow: Optical flow field [B, 2, H, W] with (u, v) components
        """
        # Normalize to [0, 1] if needed
        if image1.max() > 1.0:
            image1 = image1 / 255.0
        if image2.max() > 1.0:
            image2 = image2 / 255.0
        
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        if self.model_name.lower() == "raft":
            # RAFT requires BGR ordering internally in some versions
            flow_list = self.model(image1, image2, iters=num_iters, test_mode=True)
            flow = flow_list[-1]  # Use final prediction
        else:  # GMA
            raise NotImplementedError("Other optical flow prediction models haven't been introduced here.")
        
        return flow
    
    def compute_forward_backward_flow(self,
                                     image1: torch.Tensor,
                                     image2: torch.Tensor,
                                     num_iters: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional optical flow.
        
        Args:
            image1: Image at time t [B, C, H, W]
            image2: Image at time t+1 [B, C, H, W]
            num_iters: Number of refinement iterations
            
        Returns:
            flow_t_to_t1: Flow from t to t+1 [B, 2, H, W]
            flow_t1_to_t: Flow from t+1 to t [B, 2, H, W]
        """
        flow_t_to_t1 = self.estimate_flow(image1, image2, num_iters=num_iters)
        flow_t1_to_t = self.estimate_flow(image2, image1, num_iters=num_iters)
        
        return flow_t_to_t1, flow_t1_to_t
    
    def warp_flow(self, 
                  flow: torch.Tensor,
                  warp_flow: torch.Tensor) -> torch.Tensor:
        """
        Warp optical flow by another flow field.
        
        Used for computing backward consistency in occlusion detection.
        
        Args:
            flow: Flow field to warp [B, 2, H, W]
            warp_flow: Flow field defining warp [B, 2, H, W]
            
        Returns:
            warped_flow: Warped flow field [B, 2, H, W]
        """
        B, _, H, W = flow.shape
        
        # Create coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=flow.device),
            torch.arange(W, dtype=torch.float32, device=flow.device),
            indexing='ij'
        )
        
        # Add warp displacement
        sampled_x = grid_x + warp_flow[:, 0, :, :]  # [B, H, W]
        sampled_y = grid_y + warp_flow[:, 1, :, :]  # [B, H, W]
        
        # Normalize to [-1, 1] for grid_sample
        sampled_x = 2.0 * sampled_x / (W - 1) - 1.0
        sampled_y = 2.0 * sampled_y / (H - 1) - 1.0
        
        grid = torch.stack([sampled_x, sampled_y], dim=-1)  # [B, H, W, 2]
        
        # Warp using bilinear interpolation
        warped = F.grid_sample(flow, grid, mode='bilinear', 
                              padding_mode='zeros', align_corners=True)
        
        return warped
    
    def compute_occlusion_mask(self,
                              flow_fwd: torch.Tensor,
                              flow_bwd: torch.Tensor,
                              alpha1: float = 0.01,
                              alpha2: float = 0.5) -> torch.Tensor:
        """
        Generate occlusion mask using forward-backward consistency.
        
        Formula: ||F_fwd + F_bwd(warped)||_2 < alpha1 * ||F_fwd||_2 + alpha2
        
        Args:
            flow_fwd: Forward flow (t -> t+1) [B, 2, H, W]
            flow_bwd: Backward flow (t+1 -> t) [B, 2, H, W]
            alpha1: Relative threshold coefficient (default: 0.01)
            alpha2: Absolute threshold offset (default: 0.5)
            
        Returns:
            valid_mask: Occlusion mask [B, 1, H, W] with values in {0.0, 1.0}
                       1.0 indicates valid (non-occluded) pixels
        """
        # Warp backward flow to forward frame
        flow_bwd_warped = self.warp_flow(flow_bwd, flow_fwd)
        
        # Compute consistency error
        flow_sum = flow_fwd + flow_bwd_warped  # [B, 2, H, W]
        error = torch.norm(flow_sum, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Compute magnitude threshold
        mag_fwd = torch.norm(flow_fwd, dim=1, keepdim=True)  # [B, 1, H, W]
        threshold = alpha1 * mag_fwd + alpha2
        
        # Generate mask (1 for valid, 0 for occluded)
        valid_mask = (error < threshold).float()
        
        return valid_mask
    
    def process_sequence(self,
                        image_sequence: torch.Tensor,
                        save_dir: Optional[Path] = None,
                        alpha1: float = 0.01,
                        alpha2: float = 0.5,
                        num_iters: int = 20) -> Dict[str, torch.Tensor]:
        """
        Process a full sequence to generate flow and occlusion masks.
        
        Args:
            image_sequence: Sequence of images [T, C, H, W] or [T, H, W, C]
            save_dir: Optional directory to save flow and masks
            alpha1: Occlusion threshold coefficient
            alpha2: Occlusion threshold offset
            num_iters: RAFT refinement iterations
            
        Returns:
            Dictionary containing:
                - flow: [T-1, 2, H, W] optical flow for each frame pair
                - mask: [T-1, 1, H, W] occlusion masks
                - flow_bwd: [T-1, 2, H, W] backward flow (optional, for debugging)
        """
        # Ensure proper format [T, C, H, W]
        if image_sequence.dim() == 4 and image_sequence.shape[-1] in [1, 3, 4]:
            if image_sequence.shape[1] not in [1, 3, 4]:  # Likely [T, H, W, C]
                image_sequence = image_sequence.permute(0, 3, 1, 2)
        
        T = image_sequence.shape[0]
        flow_list = []
        mask_list = []
        
        print(f"Processing sequence of {T} frames...")
        
        for t in range(T - 1):
            img_t = image_sequence[t:t+1]  # Keep batch dimension
            img_t1 = image_sequence[t+1:t+2]
            
            print(f"  Frame {t} -> {t+1}", end="")
            
            # Compute bidirectional flow
            flow_fwd, flow_bwd = self.compute_forward_backward_flow(
                img_t, img_t1, num_iters=num_iters
            )
            
            # Compute occlusion mask
            valid_mask = self.compute_occlusion_mask(flow_fwd, flow_bwd, alpha1, alpha2)
            
            flow_list.append(flow_fwd.cpu())
            mask_list.append(valid_mask.cpu())
            
            print(f" | Mask valid pixels: {valid_mask.mean():.2%}")
        
        # Stack results
        result = {
            'flow': torch.cat(flow_list, dim=0),  # [T-1, 2, H, W]
            'mask': torch.cat(mask_list, dim=0),  # [T-1, 1, H, W]
        }
        
        # Save to disk if requested
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            np.savez(save_dir / 'flow.npz', flow=result['flow'].numpy())
            np.savez(save_dir / 'mask.npz', mask=result['mask'].numpy())
            
            print(f"Saved flow and mask to {save_dir}")
        
        return result


def load_precomputed_flow(flow_dir: Path,
                         frame_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load precomputed optical flow and occlusion mask.
    
    Args:
        flow_dir: Directory containing flow.npz and mask.npz
        frame_idx: Frame index to load
        
    Returns:
        flow: [2, H, W] optical flow
        mask: [1, H, W] occlusion mask
    """
    flow_dir = Path(flow_dir)
    
    # Load flow
    flow_data = np.load(flow_dir / 'flow.npz', allow_pickle=True)
    flow_all = torch.from_numpy(flow_data['flow']).float()  # [T, 2, H, W]
    
    # Load mask
    mask_data = np.load(flow_dir / 'mask.npz', allow_pickle=True)
    mask_all = torch.from_numpy(mask_data['mask']).float()  # [T, 1, H, W]
    
    # Extract specific frame
    flow = flow_all[frame_idx]  # [2, H, W]
    mask = mask_all[frame_idx]  # [1, H, W]
    
    return flow, mask


def visualize_flow(flow: torch.Tensor, 
                  mask: Optional[torch.Tensor] = None,
                  save_path: Optional[Path] = None) -> np.ndarray:
    """
    Visualize optical flow using HSV encoding.
    
    Args:
        flow: [2, H, W] optical flow
        mask: Optional [1, H, W] occlusion mask
        save_path: Optional path to save visualization
        
    Returns:
        Visualization as numpy array [H, W, 3] in BGR
    """
    flow_np = flow.cpu().numpy().transpose(1, 2, 0)  # [H, W, 2]
    
    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow_np[:, :, 0], flow_np[:, :, 1])
    
    # Create HSV image
    hsv = np.zeros((flow_np.shape[0], flow_np.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = np.uint8(ang * 180 / np.pi / 2)  # Hue
    hsv[:, :, 1] = 255  # Saturation
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value
    
    # Convert HSV to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply mask if provided
    if mask is not None:
        mask_np = mask[0].cpu().numpy()  # [H, W]
        bgr = bgr * mask_np[:, :, np.newaxis]
    
    if save_path is not None:
        cv2.imwrite(str(save_path), bgr)
    
    return bgr


if __name__ == "__main__":
    """
    Example usage:
    
    from utils.optic_flow_utils import OpticalFlowProcessor
    
    # Initialize processor
    processor = OpticalFlowProcessor(model_name="raft", small=False)
    
    # Load your image sequence (e.g., from NeRF-Synthetic)
    # images: torch.Tensor of shape [T, C, H, W] in range [0, 1]
    
    # Process sequence
    result = processor.process_sequence(
        images,
        save_dir="path/to/flow_data",
        alpha1=0.01,
        alpha2=0.5
    )
    
    flow = result['flow']  # [T-1, 2, H, W]
    mask = result['mask']  # [T-1, 1, H, W]
    """
    pass
