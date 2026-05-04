"""Plotting utilities for FastGS training visualisation.

All functions use the non-interactive Agg backend so they are safe to call
inside a training loop (no display required).
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for training loops
import matplotlib.pyplot as plt


def plot_cluster_gaussian_histogram(
    cluster_labels: "torch.Tensor",
    n_clusters: int,
    save_path: str,
    iteration: int,
) -> None:
    """Bar chart: number of Gaussians per cluster at a given training iteration.

    Args:
        cluster_labels: (N,) int32 tensor with values in [-1, n_clusters-1].
                        -1 denotes static Gaussians (excluded from chart).
        n_clusters:     Total number of clusters.
        save_path:      File path to save the PNG figure.
        iteration:      Current training iteration (shown in title).
    """
    import numpy as np
    labels_np = cluster_labels.cpu().numpy()
    counts = [(labels_np == k).sum() for k in range(n_clusters)]

    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 0.6), 5))
    x = np.arange(n_clusters)
    bars = ax.bar(x, counts, color="steelblue", edgecolor="white", linewidth=0.5)

    # Annotate each bar with its count
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(int(cnt)),
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Cluster Index", fontsize=11)
    ax.set_ylabel("Gaussian Count", fontsize=11)
    ax.set_title(f"Gaussians per Cluster  (iter {iteration})", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in x])
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Cluster histogram saved to {save_path}")


def plot_deformation_fft(gaussians, deform_model, save_path, use_teacher=False):
    """Plot the per-cluster mean FFT magnitude of deformation trajectories.

    Args:
        gaussians:    GaussianModel with tracked_for_fft and _cluster_labels.
        deform_model: ClusteredDeformModel (or DeformModel_4DGS) to query.
        save_path:    File path to save the PNG figure.
        use_teacher:  If True, query the teacher field; otherwise the students.
    """
    import torch
    import torch.fft

    tracked_indices = torch.nonzero(gaussians.tracked_for_fft.squeeze()).view(-1)
    if tracked_indices.numel() == 0:
        print("No tracked points for FFT.")
        return

    n_tracked = tracked_indices.shape[0]
    xyz_base = gaussians.get_xyz[tracked_indices].detach()
    cluster_labels = gaussians._cluster_labels[tracked_indices]

    n_steps = 100
    t_vals = torch.linspace(0, 1, n_steps, device="cuda")

    d_xyz_seq = torch.zeros((n_steps, n_tracked, 3), device="cuda")

    with torch.no_grad():
        for i, t in enumerate(t_vals):
            time_input = t.view(1).expand(n_tracked, 1)
            if use_teacher:
                d_xyz, _, _ = deform_model.step_teacher(xyz_base, time_input)
            else:
                d_xyz, _, _ = deform_model.step(xyz_base, time_input, cluster_ids=cluster_labels)
            d_xyz_seq[i] = d_xyz.detach()

    # FFT over the time axis (dim=0)
    d_xyz_seq_np = d_xyz_seq.cpu()
    mag_spectra = torch.abs(torch.fft.rfft(d_xyz_seq_np, dim=0))  # [n_steps//2+1, n_tracked, 3]
    mag_spectra_mean = mag_spectra.mean(dim=2).numpy()

    plt.figure(figsize=(10, 6))
    freqs = torch.fft.rfftfreq(n_steps).numpy()

    valid_clusters = torch.unique(cluster_labels)
    valid_clusters = valid_clusters[valid_clusters >= 0].cpu().numpy()
    cluster_labels_np = cluster_labels.cpu().numpy()

    for c in valid_clusters:
        mask = cluster_labels_np == c
        if mask.any():
            cluster_avg = mag_spectra_mean[:, mask].mean(axis=1)
            plt.plot(freqs, cluster_avg, label=f"Cluster {c}")

    plt.xlabel("Frequency Bin")
    plt.ylabel("FFT Magnitude (Mean per channel displacement)")
    title = "Teacher Field" if use_teacher else "Student Field"
    plt.title(f"Deformation FFT ({title})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Deformation FFT saved to {save_path}")
