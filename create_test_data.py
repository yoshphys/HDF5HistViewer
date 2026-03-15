#!/usr/bin/env python3
"""Generate a sample HDF5 file with FHist.jl-compatible histogram data for testing."""

import argparse
import numpy as np
import h5py


def write_hist1d(group, name: str, edges: np.ndarray, weights: np.ndarray):
    """Write a 1D FHist histogram into *group* under *name*."""
    assert len(weights) == len(edges) - 1
    g = group.require_group(name)
    g.create_dataset("edges_1", data=edges.astype(np.float64))
    g.create_dataset("weights", data=weights.astype(np.float64))
    g.create_dataset("sumw2",   data=weights.astype(np.float64))  # assume unit weights
    g.attrs["_producer"] = "create_test_data.py"
    g.attrs["nentries"]  = int(weights.sum())
    g.attrs["overflow"]  = False


def write_hist2d(group, name: str,
                 edges_x: np.ndarray, edges_y: np.ndarray,
                 weights: np.ndarray):
    """Write a 2D FHist histogram into *group* under *name*."""
    assert weights.shape == (len(edges_x) - 1, len(edges_y) - 1)
    g = group.require_group(name)
    g.create_dataset("edges_1", data=edges_x.astype(np.float64))
    g.create_dataset("edges_2", data=edges_y.astype(np.float64))
    g.create_dataset("weights", data=weights.astype(np.float64))
    g.create_dataset("sumw2",   data=weights.astype(np.float64))
    g.attrs["_producer"] = "create_test_data.py"
    g.attrs["nentries"]  = int(weights.sum())
    g.attrs["overflow"]  = False


def write_hist3d(group, name: str,
                 edges_x: np.ndarray, edges_y: np.ndarray, edges_z: np.ndarray,
                 weights: np.ndarray):
    """Write a 3D FHist histogram into *group* under *name*."""
    assert weights.shape == (len(edges_x) - 1, len(edges_y) - 1, len(edges_z) - 1)
    g = group.require_group(name)
    g.create_dataset("edges_1", data=edges_x.astype(np.float64))
    g.create_dataset("edges_2", data=edges_y.astype(np.float64))
    g.create_dataset("edges_3", data=edges_z.astype(np.float64))
    g.create_dataset("weights", data=weights.astype(np.float64))
    g.create_dataset("sumw2",   data=weights.astype(np.float64))
    g.attrs["_producer"] = "create_test_data.py"
    g.attrs["nentries"]  = int(weights.sum())
    g.attrs["overflow"]  = False


def create(path: str, rng: np.random.Generator):
    with h5py.File(path, "w") as f:

        # --- student/boy ---
        boy = f.require_group("student/boy")

        # height: normal distribution ~170 cm
        h_edges = np.linspace(140, 200, 31)          # 30 bins
        h_data  = rng.normal(170, 8, 500)
        h_w, _  = np.histogram(h_data, bins=h_edges)
        write_hist1d(boy, "height", h_edges, h_w.astype(float))

        # weight: normal distribution ~65 kg
        w_edges = np.linspace(40, 100, 25)
        w_data  = rng.normal(65, 10, 500)
        w_w, _  = np.histogram(w_data, bins=w_edges)
        write_hist1d(boy, "weight", w_edges, w_w.astype(float))

        # height vs weight: 2D
        hw2d, _, _ = np.histogram2d(h_data, w_data, bins=[h_edges, w_edges])
        write_hist2d(boy, "height_vs_weight", h_edges, w_edges, hw2d)

        # --- student/girl ---
        girl = f.require_group("student/girl")

        h_data_g = rng.normal(158, 7, 500)
        h_w_g, _ = np.histogram(h_data_g, bins=h_edges)
        write_hist1d(girl, "height", h_edges, h_w_g.astype(float))

        w_data_g = rng.normal(54, 8, 500)
        w_w_g, _ = np.histogram(w_data_g, bins=w_edges)
        write_hist1d(girl, "weight", w_edges, w_w_g.astype(float))

        # --- student/boy/score: 3D (height × weight × score) ---
        s_edges  = np.linspace(0, 100, 11)   # 10 bins
        s_data   = rng.uniform(40, 100, 500)
        hw3d, _  = np.histogramdd(
            np.column_stack([h_data, w_data, s_data]),
            bins=[h_edges, w_edges, s_edges],
        )
        write_hist3d(boy, "score_3d", h_edges, w_edges, s_edges, hw3d)

    print(f"Created: {path}")
    print("Structure:")
    with h5py.File(path, "r") as f:
        _print_tree(f["/"], "", True)


def _print_tree(node, prefix, is_root):
    if is_root:
        print("/")
        items = list(node.items())
        for i, (name, child) in enumerate(items):
            _print_tree(child, "", i == len(items) - 1, name)
        return

def _print_tree(node, prefix, last, name=""):
    connector = "└── " if last else "├── "
    child_prefix = prefix + ("    " if last else "│   ")
    if isinstance(node, h5py.Group) and "weights" not in node:
        print(f"{prefix}{connector}{name}/")
        items = list(node.items())
        for i, (cname, child) in enumerate(items):
            _print_tree(child, child_prefix, i == len(items) - 1, cname)
    elif isinstance(node, h5py.Group):
        # histogram: count dimensions
        ndim = sum(1 for k in node.keys() if k.startswith("edges_"))
        print(f"{prefix}{connector}{name}  [Hist{ndim}D]")
    else:
        print(f"{prefix}{connector}{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test HDF5 data for HDF5HistViewer")
    parser.add_argument("output", nargs="?", default="test_data.h5",
                        help="Output HDF5 file path (default: test_data.h5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    create(args.output, rng)
