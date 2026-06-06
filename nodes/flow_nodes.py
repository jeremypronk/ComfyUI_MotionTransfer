"""
Optical flow extraction and processing nodes.

Contains nodes for RAFT/SEA-RAFT flow extraction, flow upsampling with guided filtering,
and flow-to-STMap conversion for warping.
"""

import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional

# Import unified model loader
from ..models import OpticalFlowModel

# Import logger
from ..utils.logger import get_logger

logger = get_logger()


def run_flow_model(model, img1, img2, iters=12, test_mode=True, **kwargs):
    """
    Wrapper to safely execute optical flow models and standardize the output.
    Converts SEA-RAFT's dictionary output into a standard (flow_low, flow, uncertainty) tuple.
    """
    output = model(img1, img2, iters=iters, test_mode=test_mode, **kwargs)

    # Handle SEA-RAFT (returns a dictionary)
    if isinstance(output, dict):
        flow_low = output['flow'][0]  # First iteration (low res)
        flow_final = output['flow'][-1]  # Final iteration (high res)
        uncertainty = output['info'][-1] if 'info' in output else None
        return flow_low, flow_final, uncertainty

    # Handle standard RAFT / other models (returns a tuple or list)
    elif isinstance(output, (tuple, list)):
        if len(output) >= 3:
            return output[0], output[1], output[2]
        elif len(output) == 2:  # Some RAFTs don't return uncertainty
            return output[0], output[1], None

    # Fallback
    return output


class RAFTFlowExtractor:
    """Extract dense optical flow between consecutive frames using RAFT or SEA-RAFT.

    Supports both original RAFT (2020) and SEA-RAFT (2024 ECCV - 2.3x faster, 22% more accurate).
    Returns flow fields and confidence/uncertainty maps for motion transfer pipeline.
    """

    _model = None
    _model_path = None
    _model_type = None  # Track whether loaded model is 'raft' or 'searaft'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames from ComfyUI video loader. Expects [B, H, W, C] batch of images."
                }),
                "raft_iters": ("INT", {
                    "default": 8,
                    "min": 6,
                    "max": 32,
                    "tooltip": "Refinement iterations. SEA-RAFT needs fewer (6-8) than RAFT (12-20) for same quality. Will auto-adjust to 8 for SEA-RAFT if you leave at default 12."
                }),
                "model_name": (OpticalFlowModel.get_available_models(), {
                    "default": OpticalFlowModel.get_available_models()[1],
                    "tooltip": "Optical flow model. "
                }),
                "handle_large_motion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable multi-frame flow accumulation for large motion (v0.8+ Phase 3). Automatically subdivides frames when flow exceeds max_displacement threshold. Slower but handles fast motion better."
                }),
                "max_displacement": ("INT", {
                    "default": 128,
                    "min": 32,
                    "max": 512,
                    "tooltip": "Maximum flow magnitude (pixels) before subdivision (Phase 3). RAFT/SEA-RAFT have effective max ~256px. If motion exceeds this, frames are interpolated and flow accumulated. 128 is recommended."
                }),
            }
        }

    RETURN_TYPES = ("FLOW", "IMAGE")  # flow fields [B-1, H, W, 2], confidence [B-1, H, W, 1]
    RETURN_NAMES = ("flow", "confidence")
    FUNCTION = "extract_flow"
    CATEGORY = "MotionTransfer/Flow"

    def extract_flow(self, images, raft_iters, model_name, handle_large_motion=False, max_displacement=128):
        """Extract optical flow between consecutive frame pairs.

        Args:
            images: Tensor [B, H, W, C] in range [0, 1]
            raft_iters: Number of refinement iterations
            model_name: Model variant to use (RAFT or SEA-RAFT)
            handle_large_motion: Enable multi-frame accumulation for large motion
            max_displacement: Maximum flow magnitude before subdivision

        Returns:
            flow: Tensor [B-1, H, W, 2] containing (u, v) flow vectors
            confidence: Tensor [B-1, H, W, 1] containing flow confidence/uncertainty scores
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model (RAFT or SEA-RAFT, cached)
        model, model_type = self._load_model(model_name, device)

        # Auto-adjust iterations for SEA-RAFT if user left default
        if model_type == 'searaft' and raft_iters > 8:
            print(f"[Motion Transfer] **** WARNING recommended raft_iters is 8 for SEA-RAFT (faster convergence) you are using {raft_iters}!! ****")

        # Convert ComfyUI format [B, H, W, C] to torch [B, C, H, W]
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2).to(device)

        # Extract flow for consecutive pairs
        flows = []
        confidences = []

        with torch.no_grad():
            for i in range(len(images) - 1):
                img1 = images[i:i+1] * 255.0  # RAFT expects [0, 255]
                img2 = images[i+1:i+2] * 255.0

                # Pad to multiple of 8
                from torch.nn.functional import pad
                h, w = img1.shape[2:]
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    img1 = pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
                    img2 = pad(img2, (0, pad_w, 0, pad_h), mode='replicate')

                # Run model (RAFT or SEA-RAFT)
                flow_low, flow_up, uncertainty = run_flow_model(model, img1, img2, iters=raft_iters, test_mode=True)

                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    flow_up = flow_up[:, :, :h, :w]
                    if uncertainty is not None:
                        uncertainty = uncertainty[:, :, :h, :w]

                # Compute confidence
                if model_type == 'searaft' and uncertainty is not None:
                    # Use SEA-RAFT's native uncertainty (better than heuristic)
                    # Uncertainty is already [1, 1, H, W], convert to confidence
                    conf = 1.0 - torch.clamp(uncertainty, 0, 1)
                else:
                    # Use heuristic confidence for original RAFT
                    flow_mag = torch.sqrt(flow_up[:, 0:1]**2 + flow_up[:, 1:2]**2)
                    conf = torch.exp(-flow_mag / 10.0)

                # Check for large motion if handling is enabled
                if handle_large_motion:
                    flow_mag = torch.sqrt(flow_up[:, 0:1]**2 + flow_up[:, 1:2]**2)
                    max_motion = flow_mag.max().item()

                    if max_motion > max_displacement:
                        # Need subdivision - compute multi-frame flow
                        print(f"[RAFTFlowExtractor] Frame {i}: Large motion detected ({max_motion:.1f}px > {max_displacement}px), using multi-frame accumulation")

                        # Interpolate frames and accumulate flow
                        accumulated_flow, accumulated_conf = self._multi_frame_flow(
                            images[i:i+1], images[i+1:i+2], model, model_type,
                            raft_iters, max_displacement, device
                        )

                        flows.append(accumulated_flow)
                        confidences.append(accumulated_conf)
                        continue

                flows.append(flow_up[0].permute(1, 2, 0).cpu())  # [H, W, 2]
                confidences.append(conf[0].permute(1, 2, 0).cpu())  # [H, W, 1]

        # Stack into batch tensors
        flow_batch = torch.stack(flows, dim=0)  # [B-1, H, W, 2]
        conf_batch = torch.stack(confidences, dim=0)  # [B-1, H, W, 1]

        return (flow_batch.numpy(), conf_batch.numpy())

    @classmethod
    def _load_model(cls, model_name, device):
        """Load RAFT or SEA-RAFT model with caching.

        Uses the new unified OpticalFlowModel loader which handles both
        RAFT and SEA-RAFT models cleanly without sys.path manipulation.

        Returns:
            tuple: (model, model_type) where model_type is 'raft' or 'searaft'
        """
        if cls._model is None or cls._model_path != model_name:
            # Use the new unified loader - much simpler!
            model, model_type = OpticalFlowModel.load(model_name, device)

            cls._model = model
            cls._model_path = model_name
            cls._model_type = model_type

        return cls._model, cls._model_type

    def _multi_frame_flow(self, frame_a, frame_b, model, model_type, raft_iters, max_displacement, device):
        """Compute flow with multi-frame accumulation for large motion.

        Args:
            frame_a: [1, C, H, W] first frame
            frame_b: [1, C, H, W] second frame
            model: RAFT or SEA-RAFT model
            model_type: 'raft' or 'searaft'
            raft_iters: Refinement iterations
            max_displacement: Maximum flow magnitude before subdivision
            device: torch device

        Returns:
            accumulated_flow: [H, W, 2] total flow from frame_a to frame_b
            accumulated_conf: [H, W, 1] confidence for accumulated flow
        """
        # Estimate required subdivisions
        with torch.no_grad():
            # Quick initial flow estimate with few iterations
            _, initial_flow, _ = run_flow_model(model, frame_a * 255.0, frame_b * 255.0, iters=4, test_mode=True)

            max_motion = torch.sqrt(initial_flow[:, 0:1]**2 + initial_flow[:, 1:2]**2).max().item()
            n_subdivisions = int(np.ceil(max_motion / max_displacement))
            n_subdivisions = min(n_subdivisions, 4)  # Cap at 4 subdivisions

            print(f"  Subdividing into {n_subdivisions} intermediate frames")

            # Interpolate intermediate frames
            interp_frames = self._interpolate_frames(frame_a, frame_b, n_subdivisions, device)

            # Compute flow between each pair
            sub_flows = []
            sub_confs = []

            for j in range(len(interp_frames) - 1):
                img1 = interp_frames[j] * 255.0
                img2 = interp_frames[j+1] * 255.0

                # Pad to multiple of 8
                from torch.nn.functional import pad
                h, w = img1.shape[2:]
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    img1 = pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
                    img2 = pad(img2, (0, pad_w, 0, pad_h), mode='replicate')

                # Compute flow
                _, flow_up, uncertainty = run_flow_model(model, img1, img2, iters=raft_iters, test_mode=True)

                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    flow_up = flow_up[:, :, :h, :w]
                    if uncertainty is not None:
                        uncertainty = uncertainty[:, :, :h, :w]

                # Compute confidence
                if model_type == 'searaft' and uncertainty is not None:
                    conf = 1.0 - torch.clamp(uncertainty, 0, 1)
                else:
                    flow_mag = torch.sqrt(flow_up[:, 0:1]**2 + flow_up[:, 1:2]**2)
                    conf = torch.exp(-flow_mag / 10.0)

                sub_flows.append(flow_up)
                sub_confs.append(conf)

            # Accumulate flows
            total_flow = self._accumulate_flows(sub_flows, device)

            # Average confidences (conservative)
            avg_conf = torch.stack(sub_confs, dim=0).mean(dim=0)

            return (total_flow[0].permute(1, 2, 0).cpu(),
                    avg_conf[0].permute(1, 2, 0).cpu())

    def _interpolate_frames(self, frame_a, frame_b, n_intermediate, device):
        """Generate intermediate frames using linear interpolation.

        Args:
            frame_a: [1, C, H, W] first frame
            frame_b: [1, C, H, W] second frame
            n_intermediate: Number of intermediate frames to create
            device: torch device

        Returns:
            frames: List of [1, C, H, W] tensors including endpoints
        """
        frames = [frame_a]
        for i in range(1, n_intermediate):
            t = i / n_intermediate
            interp = frame_a * (1 - t) + frame_b * t
            frames.append(interp)
        frames.append(frame_b)
        return frames

    def _accumulate_flows(self, flows, device):
        """Accumulate multiple flow fields into single total displacement.

        Args:
            flows: List of [1, 2, H, W] flow tensors
            device: torch device

        Returns:
            total_flow: [1, 2, H, W] accumulated flow
        """
        if len(flows) == 1:
            return flows[0]

        # Start with first flow
        total = flows[0].clone()

        # Accumulate remaining flows
        for i in range(1, len(flows)):
            # Warp next flow by accumulated flow
            warped_flow = self._warp_flow_field(flows[i], total, device)
            # Add to accumulator
            total = total + warped_flow

        return total

    def _warp_flow_field(self, flow, displacement, device):
        """Warp a flow field using a displacement field.

        Args:
            flow: [1, 2, H, W] flow field to warp
            displacement: [1, 2, H, W] displacement field
            device: torch device

        Returns:
            warped_flow: [1, 2, H, W] warped flow
        """
        _, _, h, w = flow.shape

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Apply displacement
        sample_x = grid_x + displacement[0, 0, :, :]
        sample_y = grid_y + displacement[0, 1, :, :]

        # Normalize to [-1, 1] for grid_sample
        sample_x = 2.0 * sample_x / (w - 1) - 1.0
        sample_y = 2.0 * sample_y / (h - 1) - 1.0

        # Stack into grid [1, H, W, 2]
        grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)

        # Warp using grid_sample
        warped = torch.nn.functional.grid_sample(
            flow, grid, mode='bilinear', padding_mode='border', align_corners=True
        )

        return warped


# ------------------------------------------------------
# Node 2: BidirectionalFlowExtractor - Bidirectional flow with occlusion detection
# ------------------------------------------------------
class BidirectionalFlowExtractor:
    """Extract bidirectional optical flow with consistency-based occlusion detection.

    Computes both forward (frame i→i+1) and backward (frame i+1→i) flow, then checks
    for consistency to identify occluded regions and flow estimation failures. Provides
    significantly more reliable confidence maps than single-direction flow (v0.8+).
    """

    _model = None
    _model_path = None
    _model_type = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Video frames from ComfyUI video loader. Expects [B, H, W, C] batch of images."
                }),
                "raft_iters": ("INT", {
                    "default": 8,
                    "min": 6,
                    "max": 32,
                    "tooltip": "Refinement iterations. SEA-RAFT needs fewer (6-8) than RAFT (12-20) for same quality."
                }),
                "model_name": (OpticalFlowModel.get_available_models(), {
                    "default": OpticalFlowModel.get_available_models()[1],
                    "tooltip": "Optical flow model. "
                }),
                "consistency_threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "tooltip": "Forward-backward consistency error threshold. Pixels with error > threshold are marked as occluded. 1.0 is recommended."
                }),
                "adaptive_threshold": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use flow-magnitude adaptive threshold (more accurate). Recommended: True."
                }),
            }
        }

    RETURN_TYPES = ("FLOW", "FLOW", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("flow_forward", "flow_backward", "confidence", "occlusion_mask", "consistency_error")
    FUNCTION = "extract_bidirectional"
    CATEGORY = "MotionTransfer/Flow"

    def extract_bidirectional(self, images, raft_iters, model_name, consistency_threshold, adaptive_threshold):
        """Extract bidirectional flow with occlusion detection.

        Args:
            images: Tensor [B, H, W, C] in range [0, 1]
            raft_iters: Number of refinement iterations
            model_name: Model variant to use (RAFT or SEA-RAFT)
            consistency_threshold: Error threshold for occlusion detection
            adaptive_threshold: Use flow-magnitude adaptive threshold

        Returns:
            flow_forward: [B-1, H, W, 2] forward flow
            flow_backward: [B-1, H, W, 2] backward flow
            confidence: [B-1, H, W, 1] consistency-based confidence
            occlusion_mask: [B-1, H, W, 1] binary occlusion mask
            consistency_error: [B-1, H, W, 1] error magnitude visualization
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model (shared with RAFTFlowExtractor)
        model, model_type = self._load_model(model_name, device)

        # Convert ComfyUI format [B, H, W, C] to torch [B, C, H, W]
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2).to(device)

        # Extract bidirectional flow for consecutive pairs
        flows_fwd = []
        flows_bwd = []
        confidences = []
        occlusion_masks = []
        consistency_errors = []

        with torch.no_grad():
            for i in range(len(images) - 1):
                img1 = images[i:i+1] * 255.0  # RAFT expects [0, 255]
                img2 = images[i+1:i+2] * 255.0

                # Pad to multiple of 8
                from torch.nn.functional import pad
                h, w = img1.shape[2:]
                pad_h = (8 - h % 8) % 8
                pad_w = (8 - w % 8) % 8
                if pad_h > 0 or pad_w > 0:
                    img1 = pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
                    img2 = pad(img2, (0, pad_w, 0, pad_h), mode='replicate')

                # Forward flow (img1 → img2)
                flow_low_fwd, flow_fwd, uncertainty_fwd = run_flow_model(model, img1, img2, iters=raft_iters, test_mode=True)

                # Backward flow (img2 → img1)
                flow_low_bwd, flow_bwd, uncertainty_bwd = run_flow_model(model, img2, img1, iters=raft_iters, test_mode=True)

                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    flow_fwd = flow_fwd[:, :, :h, :w]
                    flow_bwd = flow_bwd[:, :, :h, :w]

                # Compute forward-backward consistency
                occlusion_mask, consistency_error = self._compute_occlusion_mask(
                    flow_fwd, flow_bwd, consistency_threshold, adaptive_threshold
                )

                # Compute confidence from consistency
                # Invert error: high consistency = high confidence
                max_error = consistency_threshold * 3  # Normalize range
                conf = 1.0 - torch.clamp(consistency_error / max_error, 0, 1)

                flows_fwd.append(flow_fwd[0].permute(1, 2, 0).cpu())  # [H, W, 2]
                flows_bwd.append(flow_bwd[0].permute(1, 2, 0).cpu())
                confidences.append(conf[0].permute(1, 2, 0).cpu())  # [H, W, 1]
                occlusion_masks.append(occlusion_mask[0].permute(1, 2, 0).cpu())
                consistency_errors.append(consistency_error[0].permute(1, 2, 0).cpu())

        # Stack into batch tensors
        flow_fwd_batch = torch.stack(flows_fwd, dim=0).numpy()  # [B-1, H, W, 2]
        flow_bwd_batch = torch.stack(flows_bwd, dim=0).numpy()
        conf_batch = torch.stack(confidences, dim=0).numpy()  # [B-1, H, W, 1]
        occl_batch = torch.stack(occlusion_masks, dim=0).numpy()
        err_batch = torch.stack(consistency_errors, dim=0).numpy()

        return (flow_fwd_batch, flow_bwd_batch, conf_batch, occl_batch, err_batch)

    def _compute_occlusion_mask(self, flow_fwd, flow_bwd, threshold, adaptive):
        """Compute forward-backward consistency and occlusion mask.

        Args:
            flow_fwd: [1, 2, H, W] forward flow
            flow_bwd: [1, 2, H, W] backward flow
            threshold: Consistency error threshold
            adaptive: Use adaptive threshold based on flow magnitude

        Returns:
            occlusion_mask: [1, 1, H, W] binary mask (1 = occluded)
            consistency_error: [1, 1, H, W] error magnitude
        """
        # Warp backward flow using forward flow
        flow_bwd_warped = self._warp_flow(flow_bwd, flow_fwd)

        # Compute consistency error: ||flow_fwd + flow_bwd_warped||
        flow_diff = flow_fwd + flow_bwd_warped
        error = torch.sqrt(flow_diff[:, 0:1]**2 + flow_diff[:, 1:2]**2)

        # Adaptive threshold based on flow magnitude
        if adaptive:
            flow_mag = torch.sqrt(flow_fwd[:, 0:1]**2 + flow_fwd[:, 1:2]**2)
            alpha = 0.01  # Scale factor
            beta = threshold  # Base threshold
            thresh = alpha * (flow_mag ** 2) + beta
        else:
            thresh = threshold

        # Occlusion mask: 1 where error > threshold
        occlusion_mask = (error > thresh).float()

        # Dilate mask to catch boundaries
        if occlusion_mask.max() > 0:
            kernel = torch.ones((1, 1, 3, 3), device=occlusion_mask.device)
            occlusion_mask = torch.nn.functional.conv2d(
                occlusion_mask, kernel, padding=1
            )
            occlusion_mask = (occlusion_mask > 0).float()

        return occlusion_mask, error

    def _warp_flow(self, flow, displacement):
        """Warp flow field using displacement field.

        Args:
            flow: [1, 2, H, W] flow field to warp
            displacement: [1, 2, H, W] displacement field

        Returns:
            warped_flow: [1, 2, H, W] warped flow
        """
        _, _, h, w = flow.shape
        device = flow.device

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # Apply displacement
        sample_x = grid_x + displacement[0, 0, :, :]
        sample_y = grid_y + displacement[0, 1, :, :]

        # Normalize to [-1, 1] for grid_sample
        sample_x = 2.0 * sample_x / (w - 1) - 1.0
        sample_y = 2.0 * sample_y / (h - 1) - 1.0

        # Stack into grid [1, H, W, 2]
        grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)

        # Warp using grid_sample
        warped = torch.nn.functional.grid_sample(
            flow, grid, mode='bilinear', padding_mode='border', align_corners=True
        )

        return warped

    @classmethod
    def _load_model(cls, model_name, device):
        """Load RAFT or SEA-RAFT model with caching (shared with RAFTFlowExtractor)."""
        if cls._model is None or cls._model_path != model_name:
            from ..models import OpticalFlowModel
            model, model_type = OpticalFlowModel.load(model_name, device)
            cls._model = model
            cls._model_path = model_name
            cls._model_type = model_type
        return cls._model, cls._model_type


# ------------------------------------------------------
# Node 3: FlowSRRefine - Upscale and refine flow fields
# ------------------------------------------------------
class FlowSRRefine:
    """Upscale and refine optical flow fields using bicubic interpolation and guided filtering.

    Upscales low-resolution flow to match high-resolution still image, with edge-aware
    smoothing to prevent flow bleeding across sharp boundaries.
    """

    _guided_filter_available = hasattr(cv2, "ximgproc") and hasattr(getattr(cv2, "ximgproc", None), "guidedFilter")
    _guided_filter_warning_shown = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW", {
                    "tooltip": "Optical flow fields from RAFTFlowExtractor. Low-resolution flow to be upscaled."
                }),
                "guide_image": ("IMAGE", {
                    "tooltip": "High-resolution still image used as guidance for edge-aware filtering. Prevents flow from bleeding across sharp edges."
                }),
                "target_width": ("INT", {
                    "default": 16000,
                    "min": 512,
                    "max": 32000,
                    "tooltip": "Target width for upscaled flow (should match your high-res still width). Common: 4K=3840, 8K=7680, 16K=15360."
                }),
                "target_height": ("INT", {
                    "default": 16000,
                    "min": 512,
                    "max": 32000,
                    "tooltip": "Target height for upscaled flow (should match your high-res still height). Common: 4K=2160, 8K=4320, 16K=8640."
                }),
                "upscale_method": (["joint_bilateral", "guided_filter"], {
                    "default": "joint_bilateral",
                    "tooltip": "Upscaling method. 'joint_bilateral': better edge preservation, prevents bleeding (v0.8+, recommended). 'guided_filter': legacy method."
                }),
                "edge_detection": (["canny", "sobel", "none"], {
                    "default": "canny",
                    "tooltip": "Edge detection method for preserving flow discontinuities. 'canny': best for sharp edges (recommended). 'sobel': gradient-based. 'none': no edge constraints."
                }),
                "edge_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Edge detection sensitivity. Lower values (0.3) detect more edges, higher values (0.7) are more selective. 0.5 is recommended."
                }),
                "guided_filter_radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "tooltip": "Radius for filtering. Larger values (16-32) give smoother flow, smaller values (4-8) preserve detail better. 8 is a good default."
                }),
                "guided_filter_eps": ("FLOAT", {
                    "default": 1e-3,
                    "min": 1e-6,
                    "max": 1.0,
                    "tooltip": "Regularization parameter. Lower values (1e-4) preserve edges better, higher values (1e-2) give smoother results. 1e-3 is recommended."
                }),
            }
        }

    RETURN_TYPES = ("FLOW",)
    RETURN_NAMES = ("flow_upscaled",)
    FUNCTION = "refine"
    CATEGORY = "MotionTransfer/Flow"

    def refine(self, flow, guide_image, target_width, target_height,
               upscale_method="joint_bilateral", edge_detection="canny", edge_threshold=0.5,
               guided_filter_radius=8, guided_filter_eps=1e-3):
        """Upscale flow fields to target resolution with edge-aware refinement.

        Args:
            flow: [B, H_lo, W_lo, 2] flow fields
            guide_image: [1, H_hi, W_hi, C] high-res still image
            target_width, target_height: Target resolution
            upscale_method: 'joint_bilateral' or 'guided_filter'
            edge_detection: 'canny', 'sobel', or 'none'
            edge_threshold: Edge detection sensitivity
            guided_filter_radius: Radius for filtering
            guided_filter_eps: Regularization parameter

        Returns:
            flow_upscaled: [B, H_hi, W_hi, 2] upscaled and refined flow
        """
        # Convert tensors to numpy arrays properly
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()
        elif not isinstance(flow, np.ndarray):
            flow = np.array(flow)

        if isinstance(guide_image, torch.Tensor):
            guide_image = guide_image.cpu().numpy()
        elif not isinstance(guide_image, np.ndarray):
            guide_image = np.array(guide_image)

        # Get guide image (use first frame if batch)
        guide = guide_image[0] if len(guide_image.shape) == 4 else guide_image

        # Resize guide to target if needed
        guide_h, guide_w = guide.shape[:2]
        if guide_h != target_height or guide_w != target_width:
            guide = cv2.resize(guide, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

        # Convert guide to grayscale for filtering
        if guide.shape[2] == 3:
            guide_gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            guide_gray = guide[:, :, 0]

        # Generate edge mask if edge detection is enabled
        edge_mask = None
        if edge_detection != "none":
            edge_mask = self._compute_edge_mask(guide, edge_detection, edge_threshold)

        # Upscale each flow field in batch
        flow_batch = flow.shape[0]
        flow_h, flow_w = flow.shape[1:3]
        scale_x = target_width / flow_w
        scale_y = target_height / flow_h

        upscaled_flows = []
        for i in range(flow_batch):
            flow_frame = flow[i]  # [H, W, 2]

            # Bicubic upscale with proper flow scaling
            flow_u = cv2.resize(flow_frame[:, :, 0], (target_width, target_height),
                              interpolation=cv2.INTER_CUBIC) * scale_x
            flow_v = cv2.resize(flow_frame[:, :, 1], (target_width, target_height),
                              interpolation=cv2.INTER_CUBIC) * scale_y

            # Apply edge-aware filtering based on method
            if upscale_method == "joint_bilateral":
                # Joint bilateral upsampling - better edge preservation
                flow_u_ref = self._joint_bilateral_upsample(
                    flow_u, guide_gray, guided_filter_radius, guided_filter_eps
                )
                flow_v_ref = self._joint_bilateral_upsample(
                    flow_v, guide_gray, guided_filter_radius, guided_filter_eps
                )
            else:
                # Legacy guided filter method
                if FlowSRRefine._guided_filter_available:
                    flow_u_ref = cv2.ximgproc.guidedFilter(
                        guide_gray, flow_u.astype(np.float32),
                        radius=guided_filter_radius, eps=guided_filter_eps
                    )
                    flow_v_ref = cv2.ximgproc.guidedFilter(
                        guide_gray, flow_v.astype(np.float32),
                        radius=guided_filter_radius, eps=guided_filter_eps
                    )
                else:
                    if not FlowSRRefine._guided_filter_warning_shown:
                        print("WARNING: opencv-contrib-python not found, using bilateral filter instead of guided filter")
                        FlowSRRefine._guided_filter_warning_shown = True
                    flow_u_ref = cv2.bilateralFilter(flow_u, guided_filter_radius, 50, 50)
                    flow_v_ref = cv2.bilateralFilter(flow_v, guided_filter_radius, 50, 50)

            # Apply edge constraints if edge mask is available
            if edge_mask is not None:
                flow_u_ref, flow_v_ref = self._apply_edge_constraints(
                    flow_u_ref, flow_v_ref, flow_u, flow_v, edge_mask
                )

            # Stack channels
            flow_refined = np.stack([flow_u_ref, flow_v_ref], axis=-1)
            upscaled_flows.append(flow_refined)

        result = np.stack(upscaled_flows, axis=0)  # [B, H_hi, W_hi, 2]
        return (result,)

    def _compute_edge_mask(self, guide, method, threshold):
        """Compute edge mask from guide image.

        Args:
            guide: [H, W, C] guide image
            method: 'canny' or 'sobel'
            threshold: Edge detection threshold

        Returns:
            edge_mask: [H, W] binary edge mask (1 = edge, 0 = smooth)
        """
        # Convert to grayscale
        if guide.shape[2] == 3:
            gray = cv2.cvtColor((guide * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (guide[:, :, 0] * 255).astype(np.uint8)

        if method == "canny":
            # Canny edge detection - best for sharp edges
            low_threshold = int(50 * threshold)
            high_threshold = int(150 * threshold)
            edges = cv2.Canny(gray, low_threshold, high_threshold)

            # Also detect edges at coarser scale
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
            edges_coarse = cv2.Canny(gray_blur, int(low_threshold * 0.7), int(high_threshold * 0.7))

            # Combine scales
            edges = np.maximum(edges, edges_coarse)

        elif method == "sobel":
            # Sobel gradient-based detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(sobelx**2 + sobely**2)
            edges = (gradient_mag > (threshold * 255)).astype(np.uint8) * 255

        else:
            return None

        # Dilate slightly to ensure edge coverage
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return (edges / 255.0).astype(np.float32)

    def _joint_bilateral_upsample(self, flow_channel, guide_gray, radius, eps):
        """Apply joint bilateral filter for edge-aware upsampling.

        Args:
            flow_channel: [H, W] flow channel (u or v)
            guide_gray: [H, W] grayscale guide image
            radius: Filter radius
            eps: Regularization parameter

        Returns:
            filtered: [H, W] filtered flow channel
        """
        # Convert to appropriate format
        guide_8bit = (guide_gray * 255).astype(np.uint8)
        flow_float = flow_channel.astype(np.float32)

        # Joint bilateral filter - uses guide for spatial weights
        # and own values for range weights
        if FlowSRRefine._guided_filter_available:
            # Use guided filter as a good approximation of joint bilateral
            filtered = cv2.ximgproc.guidedFilter(
                guide_gray, flow_float,
                radius=radius, eps=eps
            )
        else:
            # Fallback to bilateral filter
            filtered = cv2.bilateralFilter(flow_float, radius, 50, 50)

        return filtered

    def _apply_edge_constraints(self, flow_u_refined, flow_v_refined,
                                flow_u_bicubic, flow_v_bicubic, edge_mask):
        """Apply edge constraints to preserve flow discontinuities.

        At strong edges, use nearest-neighbor from bicubic to preserve
        sharp discontinuities instead of smoothed flow.

        Args:
            flow_u_refined, flow_v_refined: Filtered flow channels
            flow_u_bicubic, flow_v_bicubic: Bicubic upscaled flow channels
            edge_mask: [H, W] binary edge mask

        Returns:
            constrained_u, constrained_v: Edge-constrained flow channels
        """
        # Blend: smooth flow where no edge, sharp (bicubic) where edge
        edge_weight = edge_mask
        flow_u_constrained = flow_u_refined * (1 - edge_weight) + flow_u_bicubic * edge_weight
        flow_v_constrained = flow_v_refined * (1 - edge_weight) + flow_v_bicubic * edge_weight

        return flow_u_constrained, flow_v_constrained


# ------------------------------------------------------
# Node 4: FlowToSTMap - Convert flow to STMap for warping
# ------------------------------------------------------
class FlowToSTMap:
    """Convert optical flow (u,v) displacement fields into normalized STMap coordinates.

    STMap format: RG channels contain normalized UV coordinates [0,1] for texture lookup.
    Compatible with Nuke STMap node, After Effects RE:Map, and ComfyUI remap nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW", {
                    "tooltip": "High-resolution flow fields from FlowSRRefine. Will be converted to normalized STMap coordinates for warping."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stmap",)
    FUNCTION = "to_stmap"
    CATEGORY = "MotionTransfer/Flow"

    def to_stmap(self, flow):
        """Convert flow displacement to normalized STMap coordinates.

        Args:
            flow: [B, H, W, 2] flow fields containing (u, v) pixel displacements
                  IMPORTANT: flow[i] represents motion from frame i to frame i+1
                  For motion transfer, we need to ACCUMULATE flow to get total
                  displacement from the original still image.

        Returns:
            stmap: [B, H, W, 3] STMap with RG=normalized coords, B=unused (set to 0)
                   Format: S = (x + accumulated_u) / W, T = (y + accumulated_v) / H
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()

        batch_size, height, width, _ = flow.shape

        # Create base coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)

        # Accumulate flow vectors for motion transfer
        # flow[0] = frame0→frame1, flow[1] = frame1→frame2, etc.
        # For motion transfer from still image:
        # - Frame 0: no displacement (identity)
        # - Frame 1: flow[0]
        # - Frame 2: flow[0] + flow[1]
        # - Frame 3: flow[0] + flow[1] + flow[2]
        accumulated_flow_u = np.zeros((height, width), dtype=np.float32)
        accumulated_flow_v = np.zeros((height, width), dtype=np.float32)

        stmaps = []
        for i in range(batch_size):
            # Accumulate current flow onto total displacement
            accumulated_flow_u += flow[i, :, :, 0]
            accumulated_flow_v += flow[i, :, :, 1]

            # Compute absolute coordinates after accumulated displacement
            new_x = x_coords + accumulated_flow_u
            new_y = y_coords + accumulated_flow_v

            # Normalize to [0, 1] range for STMap
            s = new_x / (width - 1)  # Normalized S coordinate
            t = new_y / (height - 1)  # Normalized T coordinate

            # Create 3-channel STMap (RG=coords, B=unused)
            stmap = np.zeros((height, width, 3), dtype=np.float32)
            stmap[:, :, 0] = s  # R channel = S (horizontal)
            stmap[:, :, 1] = t  # G channel = T (vertical)
            stmap[:, :, 2] = 0.0  # B channel = unused

            stmaps.append(stmap)

        result = np.stack(stmaps, axis=0)  # [B, H, W, 3]
        return (result,)

