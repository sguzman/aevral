{
  description = "Aevral forecasting research development shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux = let
      pkgs = import nixpkgs { system = "x86_64-linux"; };
    in {
      default = pkgs.mkShell {
        packages = with pkgs; [ rustc cargo rustfmt clippy curl coreutils ];
      };
      svr = pkgs.mkShell {
        packages = with pkgs; [ rustc cargo rustfmt clippy coreutils libsvm ];
        AEVRAL_LIBSVM_ROOT = "${pkgs.libsvm}";
        AEVRAL_LIBSVM_VERSION = pkgs.libsvm.version;
        AEVRAL_SVM_TRAIN = "${pkgs.libsvm.bin}/bin/svm-train";
        AEVRAL_SVM_PREDICT = "${pkgs.libsvm.bin}/bin/svm-predict";
      };
      boosting = pkgs.mkShell {
        packages = with pkgs; [ rustc cargo rustfmt clippy coreutils xgboost lightgbm catboost ];
        AEVRAL_XGBOOST_LIB = "${pkgs.xgboost}/lib/libxgboost.so";
        AEVRAL_LIGHTGBM = "${pkgs.lightgbm}/bin/lightgbm";
        AEVRAL_LIGHTGBM_VERSION = pkgs.lightgbm.version;
        AEVRAL_CATBOOST = "${pkgs.catboost}/bin/catboost";
      };
    };
  };
}
