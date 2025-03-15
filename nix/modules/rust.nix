{ inputs, ... }:
{
  imports = [
    inputs.rust-flake.flakeModules.default
    inputs.rust-flake.flakeModules.nixpkgs
    inputs.process-compose-flake.flakeModule
    inputs.cargo-doc-live.flakeModule
  ];
  perSystem = { config, self', pkgs, lib, ... }: {
    rust-project.crates."ui".crane.args = {
      buildInputs = lib.optionals pkgs.stdenv.isDarwin (
        with pkgs.darwin.apple_sdk.frameworks; [
          IOKit
        ]
      );
      # ++ [
      #   pkgs.hdf5          # Добавляем HDF5
      #   pkgs.dbus          # Добавляем D-Bus
      #   pkgs.xorg.libX11   # Добавляем libX11
      #   pkgs.xorg.libXi    # Добавляем libXi
      #   pkgs.xorg.libXtst  # Добавляем libXtst
      # ];
    };

    packages.default = self'.packages.ui;
  };
}
