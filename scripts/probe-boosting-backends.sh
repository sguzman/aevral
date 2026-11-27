#!/usr/bin/env bash
set -euo pipefail
# Use the exact repository lock, never the runner's registry or ambient nixpkgs.
rev=$(jq -r .nodes.nixpkgs.locked.rev flake.lock)
source="github:NixOS/nixpkgs/$rev"
echo "backend-pin,$rev"
for name in "$1"; do
  echo "backend-start,$name"
  for attr in "$name" "python3Packages.$name"; do
    if version=$(nix eval --raw "$source#$attr.version" 2>/dev/null); then
      echo "backend-attr,$name,$attr,$version"
      # Resolve closure first; build only when available within the CI probe budget.
      if paths=$(timeout 360 nix build --no-link --print-out-paths "$source#$attr" 2>"$name-${attr//./-}.err"); then
        printf '%s\n' "$paths" | while IFS= read -r path; do
          echo "backend-output,$name,$attr,$path"
          if test -d "$path/bin"; then
            find "$path/bin" -maxdepth 1 -printf '%f\n' | sort | sed "s/^/backend-bin,$name,$attr,/"
          fi
          if test -x "$path/bin/$name"; then
            "$path/bin/$name" --version >"$name-version.txt" 2>&1 || true
            "$path/bin/$name" --help >"$name-help.txt" 2>&1 || true
            echo "backend-cli,$name,$attr,$version"
          fi
          if test -d "$path/lib"; then
            find "$path/lib" -maxdepth 3 \( -name 'libxgboost.so*' -o -name 'lib_lightgbm.so*' -o -name 'libcatboost*.so*' \) -print | sed "s/^/backend-lib,$name,$attr,/"
          fi
        done
      else
        echo "backend-build-unverified,$name,$attr"
        tail -n 12 "$name-${attr//./-}.err" || true
      fi
    else
      echo "backend-attr-unavailable,$name,$attr"
    fi
  done
  echo "backend-end,$name"
done
