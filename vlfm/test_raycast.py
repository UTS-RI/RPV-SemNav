"""
Standalone test for line-of-sight raycasting signal propagation.

Creates a synthetic obstacle map with walls and places object signals,
then visualises how raycasting propagates the signal — flat score within
line-of-sight, fully blocked by obstacles (no corner wrapping).

Usage:
    python test_raycast.py

Outputs:
    test_raycast_output.png — side-by-side comparison of:
        (1) Obstacle / free-space map
        (2) Gaussian (isotropic) signal
        (3) Raycast signal (single seed — source in free space)
        (4) Raycast signal (multi-seed — source inside obstacle)
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_obstacle_map(size: int = 200) -> np.ndarray:
    """Create a binary free-space map (1 = free, 0 = obstacle/wall).

    Layout:
        - Outer border
        - Horizontal wall in the middle with a doorway gap
        - A box obstacle in the upper area
    """
    free = np.ones((size, size), dtype=np.float32)

    # Outer walls
    free[:3, :] = 0
    free[-3:, :] = 0
    free[:, :3] = 0
    free[:, -3:] = 0

    # Horizontal wall with doorway
    wall_row = size // 2
    free[wall_row - 1 : wall_row + 2, :] = 0
    free[wall_row - 1 : wall_row + 2, 30:42] = 1  # doorway

    # Box obstacle (simulates furniture like a sofa)
    free[50:65, 120:155] = 0

    return free


def compute_gaussian(center_row, center_col, peak, sigma_px, radius_px, size):
    """Isotropic gaussian — no wall awareness."""
    signal = np.zeros((size, size), dtype=np.float32)
    r0, r1 = max(0, center_row - radius_px), min(size, center_row + radius_px + 1)
    c0, c1 = max(0, center_col - radius_px), min(size, center_col + radius_px + 1)
    rows = np.arange(r0, r1)
    cols = np.arange(c0, c1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    dist_sq = (rr - center_row) ** 2 + (cc - center_col) ** 2
    g = peak * np.exp(-dist_sq / (2.0 * sigma_px ** 2))
    g[dist_sq > radius_px ** 2] = 0.0
    signal[r0:r1, c0:c1] = g
    return signal


def compute_raycast(
    center_row: int,
    center_col: int,
    peak_value: float,
    radius_px: int,
    free_mask: np.ndarray,
    seed_radius_px: int = 20,
) -> np.ndarray:
    """Line-of-sight raycasting: flat score, blocked by obstacles.

    Multi-seed via morphological dilation when centroid is in non-free space.
    """
    size = free_mask.shape[0]
    margin = max(10, int(radius_px * 0.1))
    r0 = max(0, center_row - radius_px - margin)
    r1 = min(size, center_row + radius_px + margin + 1)
    c0 = max(0, center_col - radius_px - margin)
    c1 = min(size, center_col + radius_px + margin + 1)

    lr = center_row - r0
    lc = center_col - c0

    local_free = free_mask[r0:r1, c0:c1]
    lh, lw = local_free.shape

    # Determine seed origins
    if local_free[lr, lc] >= 0.5:
        origins = [(lr, lc)]
        print(f"  Raycast: single seed at ({lr},{lc}) [centroid in free space]")
    else:
        rows_idx = np.arange(lh)
        cols_idx = np.arange(lw)
        rr, cc = np.meshgrid(rows_idx, cols_idx, indexing="ij")
        l1_dist = np.abs(rr - lr) + np.abs(cc - lc)
        nearby_occupied = (l1_dist <= seed_radius_px) & (local_free < 0.5)

        if nearby_occupied.any():
            occupied_u8 = nearby_occupied.astype(np.uint8)
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(occupied_u8, kernel, iterations=1)
            boundary = (dilated > 0) & ~nearby_occupied & (local_free >= 0.5)
            seed_coords = np.argwhere(boundary)
            origins = [(int(s[0]), int(s[1])) for s in seed_coords]
        else:
            origins = []

        if not origins:
            print("  Raycast: no seed origins found — returning zeros")
            return np.zeros((size, size), dtype=np.float32)

        print(f"  Raycast: {len(origins)} boundary seeds [centroid in obstacle]")

    # Cast rays
    n_rays = max(720, int(2 * np.pi * radius_px * 1.5))
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    local_signal = np.zeros((lh, lw), dtype=np.float32)

    for origin_r, origin_c in origins:
        local_signal[origin_r, origin_c] = peak_value
        active = np.ones(n_rays, dtype=bool)

        for step in range(1, radius_px + 1):
            rs = np.round(origin_r + cos_a * step).astype(np.intp)
            cs = np.round(origin_c + sin_a * step).astype(np.intp)

            oob = (rs < 0) | (rs >= lh) | (cs < 0) | (cs >= lw)
            active &= ~oob
            if not active.any():
                break

            hit = np.zeros(n_rays, dtype=bool)
            valid_idx = np.where(active)[0]
            hit[valid_idx] = local_free[rs[valid_idx], cs[valid_idx]] < 0.5
            active &= ~hit

            paint_idx = np.where(active)[0]
            if paint_idx.size > 0:
                local_signal[rs[paint_idx], cs[paint_idx]] = peak_value

    signal = np.zeros((size, size), dtype=np.float32)
    signal[r0:r1, c0:c1] = local_signal
    return signal


def signal_to_rgb(signal: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    """Convert a signal map to a colour image."""
    norm = signal.copy()
    mx = norm.max()
    if mx > 0:
        norm /= mx
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    zero_free = (signal == 0) & (free_mask > 0)
    colored[zero_free] = (255, 255, 255)
    colored[free_mask < 0.5] = (60, 60, 60)
    return colored


def obstacle_map_to_rgb(free_mask: np.ndarray) -> np.ndarray:
    img = np.ones((*free_mask.shape, 3), dtype=np.uint8) * 255
    img[free_mask < 0.5] = (40, 40, 40)
    return img


def main():
    size = 200
    free_mask = make_obstacle_map(size)

    peak = 0.85
    sigma_px = 40.0
    radius_px = 120
    seed_radius_px = 20

    # ---- Test 1: source in free space (above the wall) ----
    src1_row, src1_col = 30, 60
    print(f"=== Test 1: Source in free space ({src1_row}, {src1_col}) ===")
    print(f"  free_mask at source: {free_mask[src1_row, src1_col]}")

    obs_img = obstacle_map_to_rgb(free_mask)
    cv2.circle(obs_img, (src1_col, src1_row), 4, (0, 0, 255), -1)
    cv2.putText(obs_img, "src1", (src1_col + 6, src1_row + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1)

    gauss_signal = compute_gaussian(src1_row, src1_col, peak, sigma_px, radius_px, size)
    gauss_img = signal_to_rgb(gauss_signal, free_mask)

    print("  Computing raycast (free space source)...")
    ray_signal = compute_raycast(src1_row, src1_col, peak, radius_px, free_mask, seed_radius_px)
    ray_img = signal_to_rgb(ray_signal, free_mask)

    # ---- Test 2: source inside the box obstacle (multi-seed) ----
    src2_row, src2_col = 57, 137  # inside the box obstacle
    print(f"\n=== Test 2: Source inside obstacle ({src2_row}, {src2_col}) ===")
    print(f"  free_mask at source: {free_mask[src2_row, src2_col]}")

    print("  Computing raycast (obstacle source, multi-seed)...")
    ray_multi_signal = compute_raycast(src2_row, src2_col, peak, radius_px, free_mask, seed_radius_px)
    ray_multi_img = signal_to_rgb(ray_multi_signal, free_mask)
    cv2.circle(ray_multi_img, (src2_col, src2_row), 4, (0, 0, 255), -1)

    # Labels
    for img, label in [
        (obs_img, "Obstacle Map"),
        (gauss_img, "Gaussian (isotropic)"),
        (ray_img, "Raycast (free src)"),
        (ray_multi_img, "Raycast (in obstacle)"),
    ]:
        cv2.putText(img, label, (5, size - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 0), 1, cv2.LINE_AA)

    combined = np.hstack([obs_img, gauss_img, ray_img, ray_multi_img])
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_raycast_output.png")
    cv2.imwrite(out_path, combined)
    print(f"\nSaved: {out_path}")
    print(f"Image size: {combined.shape[1]}x{combined.shape[0]}")

    # ---- Assertions ----
    # Raycast should NOT bleed through the wall
    below_wall_row = size // 2 + 30
    below_wall_col = size - 30

    ray_below = ray_signal[below_wall_row, below_wall_col]
    gauss_below = gauss_signal[below_wall_row, below_wall_col]
    print(f"\n--- Signal below wall ({below_wall_row},{below_wall_col}) ---")
    print(f"  Gaussian: {gauss_below:.6f}")
    print(f"  Raycast:  {ray_below:.6f}")

    assert ray_below == 0.0, f"Raycast should be zero below wall, got {ray_below}"
    print("  PASS: Raycast blocks signal through walls.")

    # Raycast should have NO corner wrapping (unlike FMM)
    # Check a cell that's through the doorway and around a corner
    through_door_row = size // 2 + 10
    through_door_col = 36  # straight through the doorway
    ray_through = ray_signal[through_door_row, through_door_col]
    print(f"\n--- Signal through doorway ({through_door_row},{through_door_col}) ---")
    print(f"  Raycast: {ray_through:.6f}")
    # This cell is visible through the doorway, so it should have signal
    # (rays go straight through gaps)
    if ray_through > 0:
        print("  Signal reaches straight through doorway (expected for direct LOS).")
    else:
        print("  No signal through doorway (source may be at wrong angle).")

    # Behind the L-obstacle from source 1 — should be blocked
    behind_box_row = 57
    behind_box_col = 170  # behind the box obstacle
    ray_behind = ray_signal[behind_box_row, behind_box_col]
    gauss_behind = gauss_signal[behind_box_row, behind_box_col]
    print(f"\n--- Signal behind box obstacle ({behind_box_row},{behind_box_col}) ---")
    print(f"  Gaussian: {gauss_behind:.6f}")
    print(f"  Raycast:  {ray_behind:.6f}")
    # Raycast from src1 should be blocked by the box
    assert ray_behind == 0.0, f"Raycast should be zero behind box, got {ray_behind}"
    print("  PASS: Raycast blocks signal behind obstacles.")

    # Multi-seed raycast should illuminate around the box obstacle
    # Check cells on opposite sides of the box
    left_of_box = ray_multi_signal[57, 110]   # to the left of the box
    right_of_box = ray_multi_signal[57, 165]  # to the right of the box
    above_box = ray_multi_signal[40, 137]     # above the box
    below_box = ray_multi_signal[75, 137]     # below the box (but above wall)
    print(f"\n--- Multi-seed raycast around box obstacle ---")
    print(f"  Left of box  (57,110): {left_of_box:.4f}")
    print(f"  Right of box (57,165): {right_of_box:.4f}")
    print(f"  Above box    (40,137): {above_box:.4f}")
    print(f"  Below box    (75,137): {below_box:.4f}")

    sides_illuminated = sum(1 for v in [left_of_box, right_of_box, above_box, below_box] if v > 0)
    print(f"  Sides with signal: {sides_illuminated}/4")
    assert sides_illuminated >= 3, f"Multi-seed should illuminate at least 3 sides of obstacle, got {sides_illuminated}"
    print("  PASS: Multi-seed raycast illuminates around the obstacle.")

    # Stats
    print(f"\n--- Signal stats ---")
    print(f"Gaussian:        max={gauss_signal.max():.4f}, nonzero={np.count_nonzero(gauss_signal)}")
    print(f"Raycast (free):  max={ray_signal.max():.4f}, nonzero={np.count_nonzero(ray_signal)}")
    print(f"Raycast (multi): max={ray_multi_signal.max():.4f}, nonzero={np.count_nonzero(ray_multi_signal)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
