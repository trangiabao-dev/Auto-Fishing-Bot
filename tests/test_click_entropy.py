import numpy as np
import pytest
from core.detector import TemplateMatcher, MatchResult
from core.input_sim import InputSimulator


def test_click_entropy_maximum_dispersion():
    """Validates that R2 Quasi-Monte Carlo generates non-clustering clicks across the button."""
    match = MatchResult(found=True, x=443, y=376, w=140, h=50)
    pts = []
    for _ in range(25):
        pt = TemplateMatcher.get_random_click_point(match, margin=6)
        pts.append(pt)

    # 1. All points must be within safe button bounding box
    for px, py in pts:
        assert 443 <= px <= 443 + 140
        assert 376 <= py <= 376 + 50

    # 2. Points must cover both left and right sides of the button
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = 443 + 70
    assert min(xs) < cx - 15, "Clicks must explore left half of the blue capsule"
    assert max(xs) > cx + 15, "Clicks must explore right half of the blue capsule"

    # 3. Minimum Euclidean separation between consecutive clicks
    for i in range(len(pts) - 1):
        dist = np.hypot(pts[i][0] - pts[i+1][0], pts[i][1] - pts[i+1][1])
        # Consecutive points should jump significantly across the surface
        assert dist >= 8.0, f"Clicks too close: #{i} {pts[i]} vs #{i+1} {pts[i+1]} (dist={dist:.1f})"


def test_input_sim_move_to_bezier():
    """Validates that human-curve move_to executes without exceptions."""
    sim = InputSimulator()
    # Move within primary screen coordinates
    sim.move_to(500, 400, human_curve=True)
    sim.move_to(520, 410, human_curve=True)
