{
  description = "Aevral forecasting research development shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
    in pkgs.mkShell {
      packages = with pkgs; [ rustc cargo rustfmt clippy curl coreutils ];
    };
  };
}
