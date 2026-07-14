"""
Standalone test for FMM (Fast Marching Method) flood-fill signal propagation.

Creates a synthetic obstacle map with walls and places object signals,
then visualises how FMM propagates the signal around obstacles respecting
wall geometry.

Usage:
    python test_fmm_flood.py

Outputs:
    test_fmm_flood_output.png — side-by-side comparison of:
        (1) The obstacle / free-space map
        (2) Gaussian (isotropic) signal
        (3) FMM (wall-respecting) signal
        (4) FMM no-decay (flat score within radius)
"""

import os
import sys
import numpy as np
import cv2

# Ensure the parent package is importable when running from inside vlfm/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import skfmm
    _HAS_SKFMM = True
except ImportError:
    _HAS_SKFMM = False
    print("WARNING: scikit-fmm not installed — FMM tests will fall back to gaussian.")


def make_obstacle_map(size: int = 200) -> np.ndarray:
    """Create a binary free-space map (1 = free, 0 = obstacle/wall).

    Layout:
        - Outer border (walls)
        - Horizontal wall in the middle with a doorway gap
        - L-shaped obstacle in the top-right quadrant
    """
    free = np.ones((size, size), dtype=np.float32)

    # Outer walls (3 px thick)
    free[:3, :] = 0
    free[-3:, :] = 0
    free[:, :3] = 0
    free[:, -3:] = 0

    # Horizontal wall across the middle with a small doorway
    wall_row = size // 2
    free[wall_row - 1 : wall_row + 2, :] = 0
    # Doorway: gap of 12 px near the left side
    free[wall_row - 1 : wall_row + 2, 30:42] = 1

    # L-shaped obstacle in the top-right
    free[30:70, 140:145] = 0   # vertical arm
    free[65:70, 140:180] = 0   # horizontal arm

    return free


def compute_gaussian_crop(
    center_row: int,
    center_col: int,
    peak_value: float,
    sigma_px: float,
    radius_px: int,
    size: int,
) -> np.ndarray:
    """Isotropic gaussian (no wall awareness)."""
    signal = np.zeros((size, size), dtype=np.float32)
    r0 = max(0, center_row - radius_px)
    r1 = min(size, center_row + radius_px + 1)
    c0 = max(0, center_col - radius_px)
    c1 = min(size, center_col + radius_px + 1)
    rows = np.arange(r0, r1)
    cols = np.arange(c0, c1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    dist_sq = (rr - center_row) ** 2 + (cc - center_col) ** 2
    g = peak_value * np.exp(-dist_sq / (2.0 * sigma_px ** 2))
    g[dist_sq > radius_px ** 2] = 0.0
    signal[r0:r1, c0:c1] = g
    return signal


def compute_fmm_signal(
    center_row: int,
    center_col: int,
    peak_value: float,
    sigma_px: float,
    radius_px: int,
    free_mask: np.ndarray,
    no_decay: bool = False,
    seed_radius_px: int = 20,
) -> np.ndarray:
    """FMM wall-respecting signal with morphological dilation seeding."""
    size = free_mask.shape[0]
    signal = np.zeros((size, size), dtype=np.float32)

    if not _HAS_SKFMM:
        print("skfmm not available — returning gaussian fallback")
        return compute_gaussian_crop(center_row, center_col, peak_value, sigma_px, radius_px, size)

    margin = max(10, int(radius_px * 0.1))
    r0 = max(0, center_row - radius_px - margin)
    r1 = min(size, center_row + radius_px + margin + 1)
    c0 = max(0, center_col - radius_px - margin)
    c1 = min(size, center_col + radius_px + margin + 1)

    lr = center_row - r0
    lc = center_col - c0

    local_free = free_mask[r0:r1, c0:c1]
    speed = np.where(local_free > 0, 1.0, 1e-6).astype(np.float64)
    phi = np.ones_like(speed)

    if speed[lr, lc] >= 0.5:
        phi[lr, lc] = -1.0
    else:
        # Morphological dilation boundary seeding
        rows_idx = np.arange(speed.shape[0])
        cols_idx = np.arange(speed.shape[1])
        rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
        l1_dist = np.abs(rr - lr) + np.abs(cc - lc)
        nearby_occupied = (l1_dist <= seed_radius_px) & (speed < 0.5)

        if nearby_occupied.any():
            occupied_u8 = nearby_occupied.astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(occupied_u8, kernel, iterations=1)
            boundary = (dilated > 0) & ~nearby_occupied & (speed > 0.5)
            if boundary.any():
                phi[boundary] = -1.0
            else:
                print("  No boundary seeds found — falling back to gaussian")
                return compute_gaussian_crop(center_row, center_col, peak_value, sigma_px, radius_px, size)
        else:
            print("  No nearby occupied cells — falling back to gaussian")
            return compute_gaussian_crop(center_row, center_col, peak_value, sigma_px, radius_px, size)

    try:
        travel = skfmm.travel_time(phi, speed, dx=1.0)
    except Exception as e:
        print(f"  FMM failed: {e} — falling back to gaussian")
        return compute_gaussian_crop(center_row, center_col, peak_value, sigma_px, radius_px, size)

    if no_decay:
        local_signal = np.where((travel <= radius_px) & (speed > 0.5), peak_value, 0.0)
    else:
        local_signal = peak_value * np.exp(-(travel ** 2) / (2.0 * sigma_px ** 2))
        local_signal[travel > radius_px] = 0.0
        local_signal[speed < 0.5] = 0.0

    signal[r0:r1, c0:c1] = local_signal.astype(np.float32)
    return signal


def signal_to_rgb(signal: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    """Convert a signal map to a colour image for visualisation."""
    norm = signal.copy()
    max_val = norm.max()
    if max_val > 0:
        norm /= max_val
    # Use COLORMAP_INFERNO for the signal
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    # White where signal is zero and free
    zero_free = (signal == 0) & (free_mask > 0)
    colored[zero_free] = (255, 255, 255)
    # Dark grey for obstacles
    colored[free_mask < 0.5] = (60, 60, 60)
    return colored


def obstacle_map_to_rgb(free_mask: np.ndarray) -> np.ndarray:
    """Render the obstacle map as an image."""
    img = np.ones((*free_mask.shape, 3), dtype=np.uint8) * 255
    img[free_mask < 0.5] = (40, 40, 40)
    return img


def main():
    size = 200
    free_mask = make_obstacle_map(size)

    # Signal source: placed in the top-left room (above the wall)
    src_row, src_col = 80, 80
    peak = 0.85
    sigma_px = 40.0   # ~2m at 20 px/m
    radius_px = 120    # ~6m at 20 px/m
    seed_radius_px = 20

    print(f"Map size: {size}x{size}")
    print(f"Signal source: row={src_row}, col={src_col}")
    print(f"Peak={peak}, sigma={sigma_px}px, radius={radius_px}px")
    print()

    # 1. Obstacle map
    obs_img = obstacle_map_to_rgb(free_mask)
    cv2.circle(obs_img, (src_col, src_row), 4, (0, 0, 255), -1)
    cv2.putText(obs_img, "src", (src_col + 6, src_row + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # 2. Gaussian signal
    print("Computing Gaussian signal...")
    gauss_signal = compute_gaussian_crop(src_row, src_col, peak, sigma_px, radius_px, size)
    gauss_img = signal_to_rgb(gauss_signal, free_mask)

    # 3. FMM signal (with decay)
    print("Computing FMM signal (with Gaussian decay)...")
    fmm_signal = compute_fmm_signal(
        src_row, src_col, peak, sigma_px, radius_px, free_mask,
        no_decay=False, seed_radius_px=seed_radius_px,
    )
    fmm_img = signal_to_rgb(fmm_signal, free_mask)

    # 4. FMM signal (no decay — flat)
    print("Computing FMM signal (no decay / flat)...")
    fmm_flat_signal = compute_fmm_signal(
        src_row, src_col, peak, sigma_px, radius_px, free_mask,
        no_decay=True, seed_radius_px=seed_radius_px,
    )
    fmm_flat_img = signal_to_rgb(fmm_flat_signal, free_mask)

    # Labels
    label_h = 25
    for img, label in [
        (obs_img, "Obstacle Map"),
        (gauss_img, "Gaussian (isotropic)"),
        (fmm_img, "FMM (wall-respecting)"),
        (fmm_flat_img, "FMM (no decay)"),
    ]:
        cv2.putText(img, label, (5, size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 0), 1, cv2.LINE_AA)

    # Compose side-by-side
    combined = np.hstack([obs_img, gauss_img, fmm_img, fmm_flat_img])

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_fmm_flood_output.png")
    cv2.imwrite(out_path, combined)
    print(f"\nSaved: {out_path}")
    print(f"Image size: {combined.shape[1]}x{combined.shape[0]}")

    # Also test with a source INSIDE an obstacle (common case — object centroid
    # is usually inside the detected object's bounding area)
    print("\n--- Testing source inside obstacle ---")
    src2_row, src2_col = 100, 100  # right on the horizontal wall
    print(f"Source inside wall: row={src2_row}, col={src2_col}")
    print(f"  free_mask value at source: {free_mask[src2_row, src2_col]}")

    fmm_inside_signal = compute_fmm_signal(
        src2_row, src2_col, peak, sigma_px, radius_px, free_mask,
        no_decay=False, seed_radius_px=seed_radius_px,
    )
    fmm_inside_img = signal_to_rgb(fmm_inside_signal, free_mask)
    cv2.circle(fmm_inside_img, (src2_col, src2_row), 4, (0, 0, 255), -1)
    cv2.putText(fmm_inside_img, "FMM (src in wall)", (5, size - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 0), 1, cv2.LINE_AA)

    out_path2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_fmm_flood_inside_wall.png")
    cv2.imwrite(out_path2, fmm_inside_img)
    print(f"Saved: {out_path2}")

    # Print some stats
    print(f"\n--- Signal stats ---")
    print(f"Gaussian: max={gauss_signal.max():.4f}, nonzero={np.count_nonzero(gauss_signal)}")
    print(f"FMM:      max={fmm_signal.max():.4f}, nonzero={np.count_nonzero(fmm_signal)}")
    print(f"FMM flat: max={fmm_flat_signal.max():.4f}, nonzero={np.count_nonzero(fmm_flat_signal)}")
    print(f"FMM wall: max={fmm_inside_signal.max():.4f}, nonzero={np.count_nonzero(fmm_inside_signal)}")

    # Key assertion: FMM signal should NOT bleed through the wall
    # The bottom-right corner should have no signal from FMM but may from gaussian
    corner_row, corner_col = size - 10, size - 10
    gauss_at_corner = gauss_signal[corner_row, corner_col]
    fmm_at_corner = fmm_signal[corner_row, corner_col]
    print(f"\nCorner ({corner_row},{corner_col}) — Gaussian: {gauss_at_corner:.6f}, FMM: {fmm_at_corner:.6f}")

    # FMM should be zero or near-zero in the blocked room (below wall, far side)
    below_wall_row = size // 2 + 30
    below_wall_col = size - 30
    fmm_below = fmm_signal[below_wall_row, below_wall_col]
    gauss_below = gauss_signal[below_wall_row, below_wall_col]
    print(f"Below wall ({below_wall_row},{below_wall_col}) — Gaussian: {gauss_below:.6f}, FMM: {fmm_below:.6f}")

    if _HAS_SKFMM:
        assert fmm_below < gauss_below or fmm_below < 0.01, \
            "FMM signal should not penetrate through walls!"
        print("\nPASS: FMM correctly blocks signal through walls.")
    else:
        print("\nSKIPPED FMM assertions (scikit-fmm not installed).")

    print("\nDone.")


if __name__ == "__main__":
    main()
