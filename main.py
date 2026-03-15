#!/usr/bin/env python3
"""HDF5 Histogram Viewer - CLI tool for FHist.jl HDF5 histograms using ROOT."""

import sys
import os
import h5py
import numpy as np
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

try:
    import ROOT
    ROOT.gROOT.SetBatch(False)
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    print("Warning: ROOT not available. 'draw' command will not work.", file=sys.stderr)

# Keep ROOT objects alive to prevent garbage collection
_draw_refs = []


# ---------------------------------------------------------------------------
# HDF5 / FHist helpers
# ---------------------------------------------------------------------------

def is_histogram(node) -> bool:
    """Return True if the HDF5 group is an FHist.jl histogram."""
    return (
        isinstance(node, h5py.Group)
        and "weights" in node
        and "edges_1" in node
    )


def hist_ndim(node) -> int:
    """Return the dimensionality (1–3) of an FHist histogram group."""
    for dim in (3, 2, 1):
        if f"edges_{dim}" in node:
            return dim
    return 0


# ---------------------------------------------------------------------------
# ROOT drawing
# ---------------------------------------------------------------------------

def draw_histogram(node, name: str):
    """Create and draw a ROOT histogram from an FHist.jl HDF5 group."""
    if not HAS_ROOT:
        print("ROOT is not available.")
        return

    ndim = hist_ndim(node)
    weights = node["weights"][()]
    sumw2 = node["sumw2"][()]

    if ndim == 1:
        edges = node["edges_1"][()].astype(np.float64)
        nbins = len(edges) - 1
        h = ROOT.TH1D(name, name, nbins, edges)
        h.Sumw2(False)
        for i in range(nbins):
            h.SetBinContent(i + 1, float(weights[i]))
            h.SetBinError(i + 1, float(np.sqrt(max(sumw2[i], 0.0))))
        c = ROOT.TCanvas(f"c_{name}", name, 800, 600)
        h.Draw()

    elif ndim == 2:
        edges_x = node["edges_1"][()].astype(np.float64)
        edges_y = node["edges_2"][()].astype(np.float64)
        nbins_x, nbins_y = len(edges_x) - 1, len(edges_y) - 1
        h = ROOT.TH2D(name, name, nbins_x, edges_x, nbins_y, edges_y)
        h.Sumw2(False)
        for ix in range(nbins_x):
            for iy in range(nbins_y):
                h.SetBinContent(ix + 1, iy + 1, float(weights[ix, iy]))
                h.SetBinError(ix + 1, iy + 1, float(np.sqrt(max(sumw2[ix, iy], 0.0))))
        c = ROOT.TCanvas(f"c_{name}", name, 800, 600)
        h.Draw("COLZ")

    elif ndim == 3:
        edges_x = node["edges_1"][()].astype(np.float64)
        edges_y = node["edges_2"][()].astype(np.float64)
        edges_z = node["edges_3"][()].astype(np.float64)
        nbins_x = len(edges_x) - 1
        nbins_y = len(edges_y) - 1
        nbins_z = len(edges_z) - 1
        h = ROOT.TH3D(name, name, nbins_x, edges_x, nbins_y, edges_y, nbins_z, edges_z)
        h.Sumw2(False)
        for ix in range(nbins_x):
            for iy in range(nbins_y):
                for iz in range(nbins_z):
                    h.SetBinContent(ix + 1, iy + 1, iz + 1, float(weights[ix, iy, iz]))
                    h.SetBinError(ix + 1, iy + 1, iz + 1, float(np.sqrt(max(sumw2[ix, iy, iz], 0.0))))
        c = ROOT.TCanvas(f"c_{name}", name, 800, 600)
        h.Draw("BOX")

    else:
        print(f"draw: unsupported histogram dimensionality: {ndim}")
        return

    c.Update()
    c.Draw()
    _draw_refs.extend([h, c])
    print(f"Drew Hist{ndim}D '{name}'")


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

class HDF5Completer(Completer):
    """prompt_toolkit Completer for HDF5Shell commands and paths."""

    COMMANDS = ["cd", "draw", "exit", "ls", "pwd", "tree"]
    PATH_COMMANDS = {"cd", "draw", "ls", "tree"}

    def __init__(self, shell: "HDF5Shell"):
        self.shell = shell

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        tokens = text.split()

        # Complete the command word
        if not tokens or (len(tokens) == 1 and not text.endswith(" ")):
            word = tokens[0] if tokens else ""
            for cmd in self.COMMANDS:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word), display_meta="command")
            return

        cmd = tokens[0]
        if cmd not in self.PATH_COMMANDS:
            return

        # Determine the partial path argument being typed
        arg = tokens[-1] if not text.endswith(" ") else ""

        if "/" in arg:
            sep = arg.rfind("/")
            dir_part = arg[: sep + 1]
            name_part = arg[sep + 1 :]
            base = self.shell._resolve(dir_part)
        else:
            dir_part = ""
            name_part = arg
            base = self.shell.cwd

        for child in self.shell._children(base):
            if not child.startswith(name_part):
                continue
            child_node = self.shell._node(self.shell._resolve(child, base))
            if isinstance(child_node, h5py.Group) and not is_histogram(child_node):
                display_meta = "dir"
                completion = dir_part + child + "/"
            elif is_histogram(child_node):
                ndim = hist_ndim(child_node)
                display_meta = f"Hist{ndim}D"
                completion = dir_part + child
            else:
                display_meta = ""
                completion = dir_part + child

            yield Completion(completion, start_position=-len(arg), display_meta=display_meta)


class HDF5Shell:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = h5py.File(filepath, "r")
        self.cwd = "/"

        self._session = PromptSession(
            history=InMemoryHistory(),
            completer=HDF5Completer(self),
            auto_suggest=AutoSuggestFromHistory(),
            complete_while_typing=False,  # complete only on Tab
        )

    # ------------------------------------------------------------------
    # Path utilities
    # ------------------------------------------------------------------

    def _resolve(self, path: str, base: str | None = None) -> str:
        """Resolve *path* relative to *base* (defaults to cwd)."""
        if base is None:
            base = self.cwd
        if path.startswith("/"):
            result = os.path.normpath(path)
        else:
            result = os.path.normpath(os.path.join(base, path))
        return result if result not in (".", "") else "/"

    def _node(self, path: str | None = None):
        """Return the h5py node at *path* (or cwd), None if missing."""
        p = self.cwd if path is None else path
        if p == "/":
            return self.file["/"]
        return self.file.get(p)

    def _children(self, path: str) -> list[str]:
        node = self._node(path)
        if not isinstance(node, h5py.Group):
            return []
        return list(node.keys())

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def cmd_pwd(self, _args):
        print(self.cwd)

    def cmd_ls(self, args):
        path = self._resolve(args[0]) if args else self.cwd
        node = self._node(path)
        if node is None:
            print(f"ls: {args[0]}: No such path")
            return
        if is_histogram(node):
            ndim = hist_ndim(node)
            print(f"{os.path.basename(path)}  [Hist{ndim}D]")
            return
        if not isinstance(node, h5py.Group):
            print(os.path.basename(path))
            return
        for name, child in node.items():
            if is_histogram(child):
                ndim = hist_ndim(child)
                print(f"  {name}  [Hist{ndim}D]")
            elif isinstance(child, h5py.Group):
                print(f"  {name}/")
            else:
                print(f"  {name}")

    def cmd_cd(self, args):
        if not args:
            self.cwd = "/"
            return
        dest = args[0]
        resolved = self._resolve(dest)
        node = self._node(resolved)
        if node is None:
            print(f"cd: {dest}: No such directory")
        elif is_histogram(node):
            print(f"cd: {dest}: Is a histogram, not a directory")
        elif not isinstance(node, h5py.Group):
            print(f"cd: {dest}: Not a directory")
        else:
            self.cwd = resolved

    def cmd_tree(self, args):
        path = self._resolve(args[0]) if args else self.cwd
        node = self._node(path)
        if node is None:
            print(f"tree: {args[0] if args else '.'}: No such path")
            return
        print(path)
        if isinstance(node, h5py.Group) and not is_histogram(node):
            self._tree_children(node, "")

    def _tree_children(self, group: h5py.Group, prefix: str):
        items = list(group.items())
        for idx, (name, child) in enumerate(items):
            last = idx == len(items) - 1
            connector = "└── " if last else "├── "
            child_prefix = prefix + ("    " if last else "│   ")

            if is_histogram(child):
                ndim = hist_ndim(child)
                print(f"{prefix}{connector}{name}  [Hist{ndim}D]")
            elif isinstance(child, h5py.Group):
                print(f"{prefix}{connector}{name}/")
                self._tree_children(child, child_prefix)
            else:
                print(f"{prefix}{connector}{name}")

    def cmd_draw(self, args):
        if not args:
            print("Usage: draw <histogram_path>")
            return
        path = self._resolve(args[0])
        node = self._node(path)
        if node is None:
            print(f"draw: {args[0]}: No such path")
            return
        if not is_histogram(node):
            print(f"draw: {args[0]}: Not a histogram")
            return
        draw_histogram(node, os.path.basename(path))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        print(f"HDF5 Histogram Viewer  |  {self.filepath}")
        print("Commands: exit  pwd  ls [path]  cd [path]  tree [path]  draw <path>")
        print("Tab completion, emacs keybindings, and history (C-r) are enabled.\n")

        dispatch = {
            "pwd":  self.cmd_pwd,
            "ls":   self.cmd_ls,
            "cd":   self.cmd_cd,
            "tree": self.cmd_tree,
            "draw": self.cmd_draw,
        }

        while True:
            try:
                line = self._session.prompt(
                    lambda: f"hdf5:{self.cwd}> ",
                    bottom_toolbar=lambda: f" {self.filepath}",
                ).strip()
            except KeyboardInterrupt:
                continue  # Ctrl-C clears the line, keep going
            except EOFError:
                print()
                break

            if not line:
                continue

            tokens = line.split()
            cmd, args = tokens[0], tokens[1:]

            if cmd == "exit":
                break
            elif cmd in dispatch:
                dispatch[cmd](args)
            else:
                print(f"Unknown command: {cmd}  (try: exit pwd ls cd tree draw)")

        self.file.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <file.h5>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: file not found: {filepath}")
        sys.exit(1)

    shell = HDF5Shell(filepath)
    shell.run()

    if HAS_ROOT and _draw_refs:
        try:
            PromptSession().prompt("Press Enter to exit and close all plots...")
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
