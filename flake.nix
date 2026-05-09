{
  description = "HDF5 Histogram Viewer";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python3.withPackages (ps: [
            ps.root
            ps.h5py
            ps.numpy
            ps.prompt-toolkit
            ps.pytest
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [ python ];
          };
        });
    };
}
