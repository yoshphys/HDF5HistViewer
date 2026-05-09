# HDF5HistViewer

An interactive CLI tool for browsing and visualizing histograms stored in HDF5 files produced by [FHist.jl](https://github.com/Moelf/FHist.jl). Histograms are rendered using [ROOT](https://root.cern) via its web display.

## Requirements

- [Nix](https://nixos.org/download/) with flakes enabled

All other dependencies (Python, ROOT, h5py, etc.) are managed automatically via `flake.nix`.

## Installation

Clone the repository and make the wrapper script executable:

```bash
git clone <repo-url>
cd HDF5HistViewer
chmod +x hdf5hist
```

Optionally, add the directory to your `PATH` to run `hdf5hist` from anywhere:

```bash
export PATH="$PATH:/path/to/HDF5HistViewer"
```

## Usage

```bash
./hdf5hist <file.h5>
```

On the first run, Nix downloads and builds all dependencies. Subsequent runs use the cache and start quickly.

The tool launches an interactive shell that navigates the HDF5 file as a directory tree.

### Commands

| Command | Description |
|---|---|
| `pwd` | Print the current path |
| `ls [path]` | List contents of a directory |
| `cd [path]` | Change directory (`cd` alone returns to root) |
| `tree [path]` | Show the directory tree from the given path |
| `info <path>` | Show histogram metadata and statistics |
| `draw <path>` | Draw a histogram in the browser via ROOT web display |
| `exit` | Exit the shell |

### Keyboard shortcuts

The shell uses [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) with emacs-style keybindings.

| Key | Action |
|---|---|
| `Tab` | Complete command or path |
| `Ctrl-R` | Incremental history search |
| `Ctrl-A` / `Ctrl-E` | Move to beginning / end of line |
| `Ctrl-K` | Kill to end of line |
| `Ctrl-C` | Clear current input |
| `Ctrl-D` | Exit |

Command history is persisted to `~/.local/state/hdf5histviewer/history`.

### Example session

```
HDF5 Histogram Viewer  |  data.h5
Commands: exit  pwd  ls [path]  cd [path]  tree [path]  info <path>  draw <path>

hdf5:/> tree
/
├── student/
│   ├── boy/
│   │   ├── height          [Hist1D]
│   │   ├── weight          [Hist1D]
│   │   ├── height_vs_weight  [Hist2D]
│   │   └── score_3d        [Hist3D]
│   └── girl/
│       ├── height          [Hist1D]
│       └── weight          [Hist1D]

hdf5:/> info student/boy/height
Path     : /student/boy/height
Type     : Hist1D
Integral : 500
Entries  : 500
Overflow : False
Axis X   : 30 bins  [140, 200]

Mean X  : 170.024
Std  X  : 7.983

hdf5:/> draw student/boy/height_vs_weight
Drew Hist2D 'height_vs_weight'
```

## HDF5 format

This tool reads histograms written by FHist.jl's `h5writehist`. Each histogram is stored as an HDF5 group containing:

| Dataset | Description |
|---|---|
| `edges_1` … `edges_N` | Bin edges for each axis (length = n_bins + 1) |
| `weights` | Bin counts (N-dimensional array) |
| `sumw2` | Sum of squared weights for error calculation |

Supported dimensionalities: 1D, 2D, 3D.

## Testing

Generate sample HDF5 data and run the test suite:

```bash
nix develop --command python create_test_data.py test_data.h5
nix develop --command pytest tests/ -v
```
