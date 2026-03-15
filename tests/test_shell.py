"""Unit tests for HDF5Shell (main.py)."""

import io
import os
import sys
import pytest
import numpy as np
import h5py

# ---------------------------------------------------------------------------
# Make sure the project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import main  # noqa: E402  (must come after sys.path manipulation)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def h5_path(tmp_path):
    """Create a minimal FHist-compatible HDF5 file and return its path."""
    path = tmp_path / "test.h5"
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        # student/boy/height  – Hist1D
        edges1 = np.linspace(140, 200, 11, dtype=np.float64)   # 10 bins
        w1, _  = np.histogram(rng.normal(170, 8, 200), bins=edges1)
        g = f.require_group("student/boy/height")
        g.create_dataset("edges_1", data=edges1)
        g.create_dataset("weights", data=w1.astype(float))
        g.create_dataset("sumw2",   data=w1.astype(float))

        # student/girl/weight – Hist1D
        edges_w = np.linspace(40, 100, 11, dtype=np.float64)
        w2, _   = np.histogram(rng.normal(54, 8, 200), bins=edges_w)
        g2 = f.require_group("student/girl/weight")
        g2.create_dataset("edges_1", data=edges_w)
        g2.create_dataset("weights", data=w2.astype(float))
        g2.create_dataset("sumw2",   data=w2.astype(float))

        # student/boy/height_vs_weight – Hist2D
        edges_y = np.linspace(40, 100, 6, dtype=np.float64)    # 5 bins
        w2d, _, _ = np.histogram2d(
            rng.normal(170, 8, 200), rng.normal(65, 10, 200),
            bins=[edges1, edges_y],
        )
        g3 = f.require_group("student/boy/height_vs_weight")
        g3.create_dataset("edges_1", data=edges1)
        g3.create_dataset("edges_2", data=edges_y)
        g3.create_dataset("weights", data=w2d)
        g3.create_dataset("sumw2",   data=w2d)

        # student/boy/score_3d – Hist3D
        edges_s = np.linspace(0, 100, 6, dtype=np.float64)     # 5 bins
        w3d, _  = np.histogramdd(
            np.column_stack([
                rng.normal(170, 8, 200),
                rng.normal(65, 10, 200),
                rng.uniform(0, 100, 200),
            ]),
            bins=[edges1, edges_y, edges_s],
        )
        g4 = f.require_group("student/boy/score_3d")
        g4.create_dataset("edges_1", data=edges1)
        g4.create_dataset("edges_2", data=edges_y)
        g4.create_dataset("edges_3", data=edges_s)
        g4.create_dataset("weights", data=w3d)
        g4.create_dataset("sumw2",   data=w3d)

    return str(path)


@pytest.fixture()
def shell(h5_path):
    s = main.HDF5Shell(h5_path)
    yield s
    s.file.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def capture(func, *args):
    """Call func(*args) and return stdout as a string."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        func(*args)
    finally:
        sys.stdout = old
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------

class TestResolve:
    def test_absolute(self, shell):
        assert shell._resolve("/student/boy") == "/student/boy"

    def test_relative_from_root(self, shell):
        assert shell._resolve("student") == "/student"

    def test_relative_from_subdir(self, shell):
        shell.cwd = "/student"
        assert shell._resolve("boy") == "/student/boy"

    def test_dotdot(self, shell):
        shell.cwd = "/student/boy"
        assert shell._resolve("..") == "/student"

    def test_dotdot_to_root(self, shell):
        shell.cwd = "/student"
        assert shell._resolve("..") == "/"

    def test_root_stays_root(self, shell):
        assert shell._resolve("/") == "/"


# ---------------------------------------------------------------------------
# is_histogram / hist_ndim
# ---------------------------------------------------------------------------

class TestHistHelpers:
    def test_histogram_detected(self, shell):
        node = shell._node("/student/boy/height")
        assert main.is_histogram(node)

    def test_directory_not_histogram(self, shell):
        node = shell._node("/student/boy")
        assert not main.is_histogram(node)

    def test_1d(self, shell):
        assert main.hist_ndim(shell._node("/student/boy/height")) == 1

    def test_2d(self, shell):
        assert main.hist_ndim(shell._node("/student/boy/height_vs_weight")) == 2

    def test_3d(self, shell):
        assert main.hist_ndim(shell._node("/student/boy/score_3d")) == 3


# ---------------------------------------------------------------------------
# pwd
# ---------------------------------------------------------------------------

class TestPwd:
    def test_initial(self, shell):
        out = capture(shell.cmd_pwd, [])
        assert out.strip() == "/"

    def test_after_cd(self, shell):
        shell.cwd = "/student/boy"
        out = capture(shell.cmd_pwd, [])
        assert out.strip() == "/student/boy"


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------

class TestLs:
    def test_root(self, shell):
        out = capture(shell.cmd_ls, [])
        assert "student/" in out

    def test_subdir(self, shell):
        out = capture(shell.cmd_ls, ["student"])
        assert "boy/" in out
        assert "girl/" in out

    def test_shows_hist_type(self, shell):
        out = capture(shell.cmd_ls, ["student/boy"])
        assert "[Hist1D]" in out
        assert "[Hist2D]" in out
        assert "[Hist3D]" in out

    def test_missing_path(self, shell):
        out = capture(shell.cmd_ls, ["nonexistent"])
        assert "No such path" in out


# ---------------------------------------------------------------------------
# cd
# ---------------------------------------------------------------------------

class TestCd:
    def test_cd_subdir(self, shell):
        shell.cmd_cd(["student"])
        assert shell.cwd == "/student"

    def test_cd_nested(self, shell):
        shell.cmd_cd(["student/boy"])
        assert shell.cwd == "/student/boy"

    def test_cd_absolute(self, shell):
        shell.cwd = "/student/boy"
        shell.cmd_cd(["/student/girl"])
        assert shell.cwd == "/student/girl"

    def test_cd_dotdot(self, shell):
        shell.cwd = "/student/boy"
        shell.cmd_cd([".."])
        assert shell.cwd == "/student"

    def test_cd_no_args_returns_root(self, shell):
        shell.cwd = "/student/boy"
        shell.cmd_cd([])
        assert shell.cwd == "/"

    def test_cd_into_histogram_rejected(self, shell):
        out = capture(shell.cmd_cd, ["student/boy/height"])
        assert "Is a histogram" in out
        assert shell.cwd == "/"

    def test_cd_missing_rejected(self, shell):
        out = capture(shell.cmd_cd, ["does_not_exist"])
        assert "No such directory" in out
        assert shell.cwd == "/"


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------

class TestTree:
    def test_tree_root(self, shell):
        out = capture(shell.cmd_tree, [])
        assert "student/" in out
        assert "boy/" in out
        assert "girl/" in out
        assert "[Hist1D]" in out

    def test_tree_subpath(self, shell):
        out = capture(shell.cmd_tree, ["student/boy"])
        assert "height" in out
        assert "score_3d" in out
        assert "[Hist3D]" in out

    def test_tree_missing(self, shell):
        out = capture(shell.cmd_tree, ["nowhere"])
        assert "No such path" in out


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_info_1d(self, shell):
        out = capture(shell.cmd_info, ["student/boy/height"])
        assert "Hist1D" in out
        assert "Axis X" in out
        assert "Integral" in out
        assert "Mean X" in out
        assert "Std  X" in out
        # 1D: no covariance matrix
        assert "Cov" not in out

    def test_info_2d(self, shell):
        out = capture(shell.cmd_info, ["student/boy/height_vs_weight"])
        assert "Hist2D" in out
        assert "Axis X" in out
        assert "Axis Y" in out
        assert "Mean X" in out
        assert "Mean Y" in out
        assert "Cov" in out
        assert "Corr" in out

    def test_info_3d(self, shell):
        out = capture(shell.cmd_info, ["student/boy/score_3d"])
        assert "Hist3D" in out
        assert "Axis Z" in out
        assert "Mean Z" in out
        assert "Cov" in out
        assert "Corr" in out

    def test_info_stats_values(self, shell):
        # mean of boy/height should be close to 170
        out = capture(shell.cmd_info, ["student/boy/height"])
        for line in out.splitlines():
            if line.startswith("Mean X"):
                mean_val = float(line.split(":")[1].strip())
                assert 160 < mean_val < 180
                break

    def test_info_not_histogram(self, shell):
        out = capture(shell.cmd_info, ["student/boy"])
        assert "Not a histogram" in out

    def test_info_missing(self, shell):
        out = capture(shell.cmd_info, ["nowhere"])
        assert "No such path" in out


# ---------------------------------------------------------------------------
# draw (ROOT-free)
# ---------------------------------------------------------------------------

class TestDraw:
    def test_draw_no_root(self, shell, monkeypatch):
        monkeypatch.setattr(main, "HAS_ROOT", False)
        out = capture(shell.cmd_draw, ["student/boy/height"])
        assert "ROOT is not available" in out

    def test_draw_missing(self, shell):
        out = capture(shell.cmd_draw, ["student/boy/missing"])
        assert "No such path" in out

    def test_draw_not_histogram(self, shell):
        out = capture(shell.cmd_draw, ["student/boy"])
        assert "Not a histogram" in out

    def test_draw_no_args(self, shell):
        out = capture(shell.cmd_draw, [])
        assert "Usage" in out
