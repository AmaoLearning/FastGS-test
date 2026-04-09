"""Unit tests for frequency-based capacity allocation.

Tests the three new functions introduced by plan_frequency_capacity.md:
  1. analyze_cluster_capacity_needs  – temporal complexity & heterogeneity
  2. allocate_capacity_by_frequency  – independent HexPlane / MLP tier assignment
  3. End-to-end compatibility with ClusteredDeformModel.__init__
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch

# Ensure the project root is importable.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.cluster_utils import (
    allocate_capacity_by_frequency,
    analyze_cluster_capacity_needs,
)

# ---------------------------------------------------------------------------
# Helpers: mock objects so we don't need real GaussianModel / DeformModel
# ---------------------------------------------------------------------------

class _FakeGaussians:
    """Minimal mock that provides ``get_xyz``."""

    def __init__(self, xyz: torch.Tensor) -> None:
        self._xyz = xyz

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz


class _SyntheticDeform:
    """Return analytic displacements so tests can control the frequency content.

    Each Gaussian is assigned a frequency band via *per_gaussian_freq*.
    Displacement at time *t* for Gaussian *i*:
        d_xyz[i] = amplitude * sin(2π * freq[i] * t)
    """

    def __init__(
        self,
        per_gaussian_freq: torch.Tensor,
        amplitude: float = 0.01,
    ) -> None:
        self.freq = per_gaussian_freq   # (N,)
        self.amp = amplitude

    def step(
        self, xyz: torch.Tensor, time_emb: torch.Tensor
    ):
        t = time_emb[:, 0]  # (N,)
        phase = 2.0 * math.pi * self.freq * t       # (N,)
        d = self.amp * torch.sin(phase).unsqueeze(-1).expand_as(xyz)  # (N, 3)
        zero = torch.zeros_like(xyz[:, :4])  # dummy rotation
        return d, zero, torch.zeros_like(xyz)


class _HeterogeneousDeform:
    """Deform model where each Gaussian has its own independent trajectory.

    Gaussian *i* moves along a random direction scaled by *per_gaussian_speed*.
    """

    def __init__(self, directions: torch.Tensor, speeds: torch.Tensor) -> None:
        self.dirs = directions   # (N, 3)
        self.speeds = speeds     # (N,)

    def step(self, xyz: torch.Tensor, time_emb: torch.Tensor):
        t = time_emb[:, 0]                                          # (N,)
        d = self.dirs * (self.speeds * t).unsqueeze(-1)              # (N, 3)
        return d, torch.zeros_like(xyz[:, :4]), torch.zeros_like(xyz)


# ---------------------------------------------------------------------------
# Test 1: Temporal complexity correctly ranks high-freq > low-freq
# ---------------------------------------------------------------------------

def test_temporal_complexity_ranking() -> None:
    """Clusters with higher oscillation frequency must have larger complexity."""
    N_per_cluster = 200
    n_clusters = 3
    N = N_per_cluster * n_clusters
    device = "cpu"

    # Cluster 0: low freq (1 Hz), Cluster 1: medium (5 Hz), Cluster 2: high (20 Hz)
    freqs = torch.cat([
        torch.full((N_per_cluster,), 1.0),
        torch.full((N_per_cluster,), 5.0),
        torch.full((N_per_cluster,), 20.0),
    ])
    labels = torch.cat([
        torch.full((N_per_cluster,), 0, dtype=torch.int32),
        torch.full((N_per_cluster,), 1, dtype=torch.int32),
        torch.full((N_per_cluster,), 2, dtype=torch.int32),
    ])

    xyz = torch.randn(N, 3)
    gaussians = _FakeGaussians(xyz)
    deform = _SyntheticDeform(freqs, amplitude=0.01)

    result = analyze_cluster_capacity_needs(
        gaussians, deform, labels, n_clusters, n_time_samples=64,
    )

    tc = result["temporal_complexity"]
    print(f"[TEST 1] temporal_complexity = {tc}")
    assert tc[2] > tc[1] > tc[0], (
        f"Expected cluster 2 > 1 > 0, got {tc}"
    )
    print("[TEST 1] PASSED: high-freq cluster has highest temporal complexity.\n")


# ---------------------------------------------------------------------------
# Test 2: Constant velocity → zero acceleration → near-zero complexity
# ---------------------------------------------------------------------------

def test_constant_velocity_zero_complexity() -> None:
    """Uniform linear motion should yield near-zero temporal complexity."""
    N = 100
    n_clusters = 1
    labels = torch.zeros(N, dtype=torch.int32)
    xyz = torch.randn(N, 3)

    # All Gaussians move at constant velocity (freq = 0 → sin(0) = 0)
    # Use a custom deform that returns d = v * t (linear)
    class _LinearDeform:
        def step(self, xyz, time_emb):
            t = time_emb[:, 0:1]  # (N, 1)
            d = 0.1 * t.expand_as(xyz)  # constant velocity
            return d, torch.zeros_like(xyz[:, :4]), torch.zeros_like(xyz)

    gaussians = _FakeGaussians(xyz)
    deform = _LinearDeform()

    result = analyze_cluster_capacity_needs(
        gaussians, deform, labels, n_clusters, n_time_samples=32,
    )

    tc = result["temporal_complexity"][0]
    print(f"[TEST 2] temporal_complexity for linear motion = {tc:.8f}")
    assert tc < 1e-5, f"Expected near-zero complexity, got {tc}"
    print("[TEST 2] PASSED: constant velocity → near-zero complexity.\n")


# ---------------------------------------------------------------------------
# Test 3: Heterogeneity – rigid vs. diverse trajectories
# ---------------------------------------------------------------------------

def test_heterogeneity_ranking() -> None:
    """A cluster with diverse trajectories should have higher heterogeneity
    than a cluster where all Gaussians move identically (rigid body)."""
    N_per = 200
    n_clusters = 2
    N = N_per * n_clusters
    device = "cpu"

    # Cluster 0: all Gaussians move identically (rigid body)
    dirs_rigid = torch.tensor([1.0, 0.0, 0.0]).unsqueeze(0).expand(N_per, 3)
    speeds_rigid = torch.ones(N_per) * 0.1

    # Cluster 1: each Gaussian has a random direction and speed
    torch.manual_seed(42)
    dirs_diverse = torch.randn(N_per, 3)
    dirs_diverse = dirs_diverse / dirs_diverse.norm(dim=-1, keepdim=True)
    speeds_diverse = torch.rand(N_per) * 0.2

    dirs = torch.cat([dirs_rigid, dirs_diverse], dim=0)
    speeds = torch.cat([speeds_rigid, speeds_diverse], dim=0)

    labels = torch.cat([
        torch.full((N_per,), 0, dtype=torch.int32),
        torch.full((N_per,), 1, dtype=torch.int32),
    ])

    xyz = torch.randn(N, 3)
    gaussians = _FakeGaussians(xyz)
    deform = _HeterogeneousDeform(dirs, speeds)

    result = analyze_cluster_capacity_needs(
        gaussians, deform, labels, n_clusters, n_time_samples=16,
    )

    het = result["heterogeneity"]
    print(f"[TEST 3] heterogeneity = {het}")
    assert het[1] > het[0], (
        f"Expected diverse cluster (1) > rigid cluster (0), got {het}"
    )
    print("[TEST 3] PASSED: diverse cluster has higher heterogeneity.\n")


# ---------------------------------------------------------------------------
# Test 4: allocate_capacity_by_frequency output format
# ---------------------------------------------------------------------------

def test_allocate_capacity_format() -> None:
    """Verify that allocate_capacity_by_frequency produces configs compatible
    with ClusteredDeformModel.__init__ (required keys present, correct types).
    MLP hidden dim is derived from hex tier (no independent MLP axis)."""
    config_path = os.path.join(_PROJECT_ROOT, "arguments", "capacity_tier_configs.json")
    with open(config_path, "r") as f:
        tier_configs = json.load(f)

    n_clusters = 6
    tc = [0.1, 0.5, 0.9, 0.3, 0.7, 0.05]
    het = [0.8, 0.2, 0.4, 0.1, 0.6, 0.9]

    configs = allocate_capacity_by_frequency(tc, het, n_clusters, tier_configs)

    assert len(configs) == n_clusters

    required_keys = {
        "spatial_resolutions", "time_resolutions", "feat_dim",
        "mlp_hidden_dim", "mlp_layer_num", "hex_tier", "mlp_tier", "tier",
    }
    for k, cfg in enumerate(configs):
        missing = required_keys - set(cfg.keys())
        assert not missing, f"Cluster {k} missing keys: {missing}"
        assert isinstance(cfg["spatial_resolutions"], list)
        assert isinstance(cfg["time_resolutions"], list)
        assert isinstance(cfg["feat_dim"], int)
        assert isinstance(cfg["mlp_hidden_dim"], int)
        assert cfg["hex_tier"] in ("high", "medium", "low")
        # MLP tier mirrors hex tier (derived, not independent)
        assert cfg["mlp_tier"] == cfg["hex_tier"], (
            f"Cluster {k}: mlp_tier should mirror hex_tier, "
            f"got mlp_tier={cfg['mlp_tier']}, hex_tier={cfg['hex_tier']}"
        )

    print(f"[TEST 4] configs = ")
    for k, cfg in enumerate(configs):
        print(f"  cluster {k}: hex={cfg['hex_tier']}, "
              f"spatial={cfg['spatial_resolutions']}, feat={cfg['feat_dim']}, "
              f"mlp_hidden={cfg['mlp_hidden_dim']}")

    # The cluster with highest temporal complexity should get hex_tier=high
    max_tc_idx = tc.index(max(tc))
    assert configs[max_tc_idx]["hex_tier"] == "high", (
        f"Cluster {max_tc_idx} has highest complexity but hex_tier={configs[max_tc_idx]['hex_tier']}"
    )

    print("[TEST 4] PASSED: output format correct & tier ranking consistent.\n")


# ---------------------------------------------------------------------------
# Test 5: Independent tier assignment (high hex + low mlp is possible)
# ---------------------------------------------------------------------------

def test_derived_mlp_from_hex() -> None:
    """MLP hidden dim is derived from hex tier's decoder input dim.
    Verify: mlp_hidden = ceil(decoder_in × 0.5) for each hex tier."""

    config_path = os.path.join(_PROJECT_ROOT, "arguments", "capacity_tier_configs.json")
    with open(config_path, "r") as f:
        tier_configs = json.load(f)

    n_clusters = 3
    # Cluster 0: highest complexity → hex=high
    # Cluster 1: lowest complexity  → hex=low
    # Cluster 2: medium             → hex=medium
    tc = [0.9, 0.1, 0.5]
    het = [0.1, 0.9, 0.5]  # heterogeneity should NOT affect MLP

    configs = allocate_capacity_by_frequency(tc, het, n_clusters, tier_configs)

    # Expected derived MLP hidden dims (ratio=0.5):
    #   high:   ceil((3*6*12 + 76) * 0.5) = ceil(292 * 0.5) = 146
    #   low:    ceil((2*6*4  + 76) * 0.5) = ceil(124 * 0.5) = 62
    #   medium: ceil((3*6*8  + 76) * 0.5) = ceil(220 * 0.5) = 110
    expected = {
        "high": math.ceil(292 * 0.5),    # 146
        "medium": math.ceil(220 * 0.5),  # 110
        "low": math.ceil(124 * 0.5),     # 62
    }

    print(f"[TEST 5] Derived MLP from hex tier:")
    for k, cfg in enumerate(configs):
        h = cfg["hex_tier"]
        print(f"  cluster {k}: hex_tier={h}, mlp_hidden={cfg['mlp_hidden_dim']} (expected={expected[h]})")
        assert cfg["mlp_tier"] == h, (
            f"Cluster {k}: mlp_tier should equal hex_tier, "
            f"got mlp_tier={cfg['mlp_tier']}, hex_tier={h}"
        )
        assert cfg["mlp_hidden_dim"] == expected[h], (
            f"Cluster {k}: hex={h} → expected mlp_hidden={expected[h]}, "
            f"got {cfg['mlp_hidden_dim']}"
        )

    print("[TEST 5] PASSED: MLP hidden dim correctly derived from hex tier.\n")


# ---------------------------------------------------------------------------
# Test 6: infer_student_configs_from_weights with dual-tier entries
# ---------------------------------------------------------------------------

def test_infer_configs_dual_tier() -> None:
    """Verify that infer_student_configs_from_weights correctly reconstructs
    student configs from dual-tier (frequency-based) weight filenames.
    MLP hidden dim should be derived from hex tier, ignoring the mlp_tier
    in the filename."""
    from utils.cluster_utils import infer_student_configs_from_weights

    config_path = os.path.join(_PROJECT_ROOT, "arguments", "capacity_tier_configs.json")
    with open(config_path, "r") as f:
        tier_configs = json.load(f)

    n_clusters = 3
    # Simulate cluster_tiers parsed from dual-tier filenames
    # (old files may have different mlp_tier; inference should derive from hex)
    cluster_tiers = {
        0: {"hex_tier": "high", "mlp_tier": "low"},
        1: {"hex_tier": "low", "mlp_tier": "high"},
        2: {"hex_tier": "medium", "mlp_tier": "medium"},
    }

    configs = infer_student_configs_from_weights(cluster_tiers, n_clusters, tier_configs)

    # Expected derived MLP hidden (ratio=0.5):
    #   high → ceil(292 * 0.5) = 146,  low → ceil(124 * 0.5) = 62,  medium → ceil(220 * 0.5) = 110
    assert len(configs) == n_clusters
    # Cluster 0: high hex → 3 spatial levels, derived mlp_hidden=146
    assert len(configs[0]["spatial_resolutions"]) == 3, configs[0]["spatial_resolutions"]
    assert configs[0]["mlp_hidden_dim"] == 146, (
        f"Cluster 0: hex=high → expected mlp_hidden=146, got {configs[0]['mlp_hidden_dim']}"
    )
    assert configs[0]["hex_tier"] == "high"
    assert configs[0]["mlp_tier"] == "high"  # mirrors hex tier

    # Cluster 1: low hex → 2 spatial levels, derived mlp_hidden=62
    assert len(configs[1]["spatial_resolutions"]) == 2, configs[1]["spatial_resolutions"]
    assert configs[1]["mlp_hidden_dim"] == 62, (
        f"Cluster 1: hex=low → expected mlp_hidden=62, got {configs[1]['mlp_hidden_dim']}"
    )

    print("[TEST 6] configs = ")
    for k, cfg in enumerate(configs):
        print(f"  cluster {k}: hex={cfg['hex_tier']}, mlp_tier={cfg['mlp_tier']}, "
              f"spatial={cfg['spatial_resolutions']}, mlp_hidden={cfg['mlp_hidden_dim']}")
    print("[TEST 6] PASSED: dual-tier config inference with derived MLP works correctly.\n")


# ---------------------------------------------------------------------------
# Test 7: infer_student_configs_from_weights with single-tier entries (backward compat)
# ---------------------------------------------------------------------------

def test_infer_configs_single_tier() -> None:
    """Verify backward compatibility: single-tier string entries still work."""
    from utils.cluster_utils import infer_student_configs_from_weights

    config_path = os.path.join(_PROJECT_ROOT, "arguments", "capacity_tier_configs.json")
    with open(config_path, "r") as f:
        tier_configs = json.load(f)

    n_clusters = 3
    cluster_tiers = {0: "high", 1: "medium", 2: "low"}

    configs = infer_student_configs_from_weights(cluster_tiers, n_clusters, tier_configs)
    assert len(configs) == n_clusters
    assert configs[0]["tier"] == "high"
    assert configs[1]["tier"] == "medium"
    assert configs[2]["tier"] == "low"

    print("[TEST 7] PASSED: single-tier config inference backward compatible.\n")


# ---------------------------------------------------------------------------
# Test 8: SNER — known high-frequency residual should yield SNER ≫ 0
# ---------------------------------------------------------------------------

def test_sner_high_residual() -> None:
    """When teacher oscillates at 20 Hz and student is static, SNER should be
    non-trivially positive (most energy above a low Nyquist cutoff)."""
    from utils.cluster_utils import compute_sner_per_cluster

    N = 200
    n_clusters = 1
    labels = torch.zeros(N, dtype=torch.int32)
    xyz = torch.randn(N, 3)
    gaussians = _FakeGaussians(xyz)

    # Teacher: 20 Hz oscillation; Student: static (zero displacement)
    class _HighFreqTeacher:
        def step_teacher(self, xyz, time_emb):
            t = time_emb[:, 0]
            d = 0.01 * torch.sin(2 * math.pi * 20.0 * t).unsqueeze(-1).expand_as(xyz)
            return d, torch.zeros_like(xyz[:, :4]), torch.zeros_like(xyz)

        def step(self, xyz, time_emb, labels):
            return torch.zeros_like(xyz), torch.zeros_like(xyz[:, :4]), torch.zeros_like(xyz)

        student_configs = [{"time_resolutions": [8]}]  # Nyquist = 4

    deform = _HighFreqTeacher()
    result = compute_sner_per_cluster(gaussians, deform, labels, n_clusters, n_time_samples=64)

    sner = result["sner"][0]
    rmse = result["distill_rmse"][0]
    print(f"[TEST 8] SNER = {sner:.4f}, RMSE = {rmse:.6f}")
    assert sner > 0.1, f"Expected SNER > 0.1 for high-freq teacher, got {sner}"
    assert rmse > 0.0, f"Expected non-zero RMSE, got {rmse}"
    print("[TEST 8] PASSED: high-frequency residual → high SNER.\n")


# ---------------------------------------------------------------------------
# Test 9: SNER — identical teacher & student → SNER ≈ 0
# ---------------------------------------------------------------------------

def test_sner_zero_residual() -> None:
    """When teacher and student produce identical displacements, SNER and
    RMSE should both be approximately zero."""
    from utils.cluster_utils import compute_sner_per_cluster

    N = 100
    n_clusters = 1
    labels = torch.zeros(N, dtype=torch.int32)
    xyz = torch.randn(N, 3)
    gaussians = _FakeGaussians(xyz)

    class _IdenticalDeform:
        def step_teacher(self, xyz, time_emb):
            t = time_emb[:, 0]
            d = 0.01 * torch.sin(2 * math.pi * 3.0 * t).unsqueeze(-1).expand_as(xyz)
            return d, torch.zeros_like(xyz[:, :4]), torch.zeros_like(xyz)

        def step(self, xyz, time_emb, labels):
            t = time_emb[:, 0]
            d = 0.01 * torch.sin(2 * math.pi * 3.0 * t).unsqueeze(-1).expand_as(xyz)
            return d, torch.zeros_like(xyz[:, :4]), torch.zeros_like(xyz)

        student_configs = [{"time_resolutions": [64]}]

    deform = _IdenticalDeform()
    result = compute_sner_per_cluster(gaussians, deform, labels, n_clusters, n_time_samples=64)

    sner = result["sner"][0]
    rmse = result["distill_rmse"][0]
    print(f"[TEST 9] SNER = {sner:.6f}, RMSE = {rmse:.8f}")
    assert sner < 1e-6, f"Expected SNER ≈ 0, got {sner}"
    assert rmse < 1e-6, f"Expected RMSE ≈ 0, got {rmse}"
    print("[TEST 9] PASSED: identical teacher & student → zero SNER.\n")


# ---------------------------------------------------------------------------
# Test 10: MLP effective rank — full-rank vs low-rank activations
# ---------------------------------------------------------------------------

def test_mlp_effective_rank() -> None:
    """A student with diverse activations should have higher effective rank
    than one producing near-constant outputs."""
    from utils.cluster_utils import compute_mlp_effective_rank

    N = 300
    n_clusters = 2
    labels = torch.cat([
        torch.zeros(N // 2, dtype=torch.int32),
        torch.ones(N // 2, dtype=torch.int32),
    ])
    xyz = torch.randn(N, 3)
    gaussians = _FakeGaussians(xyz)

    hidden_dim = 32

    # Build a minimal mock ClusteredDeformModel with two students.
    # Student 0: identity-like (diverse activations → high rank)
    # Student 1: near-constant output (low rank)
    import torch.nn as nn

    class _MockDecoder(nn.Module):
        def __init__(self, use_identity: bool) -> None:
            super().__init__()
            in_dim = 3 + 1  # xyz + time (simplified)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 10),
            )
            if not use_identity:
                # Collapse weights to produce near-constant output
                with torch.no_grad():
                    self.net[0].weight.zero_()
                    self.net[0].bias.fill_(1.0)

    class _MockStudent(nn.Module):
        def __init__(self, use_identity: bool) -> None:
            super().__init__()
            self.decoder = _MockDecoder(use_identity)

        def forward(self, xyz, time_emb):
            x = torch.cat([xyz, time_emb], dim=-1)
            return self.decoder.net(x)

    class _MockClusteredDeform:
        students = nn.ModuleList([
            _MockStudent(use_identity=True),
            _MockStudent(use_identity=False),
        ])

    deform = _MockClusteredDeform()
    result = compute_mlp_effective_rank(deform, gaussians, labels, n_clusters,
                                       n_time_samples=8, max_gaussians_per_cluster=200)

    rank_diverse = result["effective_rank"][0]
    rank_constant = result["effective_rank"][1]
    util_diverse = result["utilisation"][0]
    util_constant = result["utilisation"][1]

    print(f"[TEST 10] Diverse student:  eff_rank={rank_diverse:.2f}, utilisation={util_diverse:.3f}")
    print(f"[TEST 10] Constant student: eff_rank={rank_constant:.2f}, utilisation={util_constant:.3f}")

    assert rank_diverse > rank_constant, (
        f"Expected diverse ({rank_diverse:.2f}) > constant ({rank_constant:.2f})"
    )
    assert util_diverse > util_constant, (
        f"Expected diverse utilisation ({util_diverse:.3f}) > constant ({util_constant:.3f})"
    )
    print("[TEST 10] PASSED: diverse activations → higher effective rank.\n")


# ---------------------------------------------------------------------------
# Test 11: MLP hidden dim floor — high hex + low mlp must be bumped
# ---------------------------------------------------------------------------

def test_mlp_derived_custom_ratio() -> None:
    """Verify that mlp_ratio parameter correctly scales the derived MLP hidden dim.
    With ratio=0.75, mlp_hidden should be ceil(decoder_in × 0.75)."""
    config_path = os.path.join(_PROJECT_ROOT, "arguments", "capacity_tier_configs.json")
    with open(config_path, "r") as f:
        tier_configs = json.load(f)

    n_clusters = 3
    tc = [0.9, 0.1, 0.5]
    het = [0.1, 0.9, 0.5]

    configs = allocate_capacity_by_frequency(
        tc, het, n_clusters, tier_configs, mlp_ratio=0.75,
    )

    # Expected with ratio=0.75:
    #   high:   ceil(292 * 0.75) = ceil(219.0) = 219
    #   low:    ceil(124 * 0.75) = ceil(93.0) = 93
    #   medium: ceil(220 * 0.75) = ceil(165.0) = 165
    expected = {
        "high": math.ceil(292 * 0.75),    # 219
        "medium": math.ceil(220 * 0.75),  # 165
        "low": math.ceil(124 * 0.75),     # 93
    }

    print(f"[TEST 11] Derived MLP with ratio=0.75:")
    for k, cfg in enumerate(configs):
        h = cfg["hex_tier"]
        print(f"  cluster {k}: hex_tier={h}, mlp_hidden={cfg['mlp_hidden_dim']} (expected={expected[h]})")
        assert cfg["mlp_hidden_dim"] == expected[h], (
            f"Cluster {k}: hex={h}, ratio=0.75 → expected {expected[h]}, "
            f"got {cfg['mlp_hidden_dim']}"
        )

    print("[TEST 11] PASSED: custom mlp_ratio correctly applied.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Frequency-based Capacity Allocation — Unit Tests")
    print("=" * 60 + "\n")

    test_temporal_complexity_ranking()
    test_constant_velocity_zero_complexity()
    test_heterogeneity_ranking()
    test_allocate_capacity_format()
    test_derived_mlp_from_hex()
    test_infer_configs_dual_tier()
    test_infer_configs_single_tier()
    test_sner_high_residual()
    test_sner_zero_residual()
    test_mlp_effective_rank()
    test_mlp_derived_custom_ratio()

    print("=" * 60)
    print("ALL 11 TESTS PASSED")
    print("=" * 60)
