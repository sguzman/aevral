#!/usr/bin/env bash
set -euo pipefail
printf 'libsvm-version,%s\n' "$AEVRAL_LIBSVM_VERSION"
printf 'libsvm-root,%s\n' "$AEVRAL_LIBSVM_ROOT"
for p in svm-train svm-predict; do
  if command -v "$p" >/dev/null; then printf 'libsvm-tool,%s,%s\n' "$p" "$(command -v "$p")"; else printf 'libsvm-tool,%s,missing\n' "$p"; fi
done
find "$AEVRAL_LIBSVM_ROOT" -maxdepth 4 \( -name 'libsvm.so*' -o -name 'libsvm.a' -o -name 'svm.h' -o -name '*svm-train*' -o -name '*svm-predict*' \) -print | sort | sed 's/^/libsvm-file,/'
if command -v svm-train >/dev/null && command -v svm-predict >/dev/null; then
  dir="$(mktemp -d)"
  trap 'rm -rf "$dir"' EXIT
  printf '%s\n' '-2 1:-2' '-1 1:-1' '0 1:0' '1 1:1' '2 1:2' > "$dir/train"
  printf '%s\n' '0 1:-1' '0 1:1' > "$dir/eval"
  svm-train -s 3 -t 2 -g 0.25 -c 10 -p 0.05 -e 0.001 -h 1 -m 128 -b 0 "$dir/train" "$dir/model"
  svm-predict -b 0 "$dir/eval" "$dir/model" "$dir/pred"
  sed -n '1,/^SV$/p' "$dir/model" | sed 's/^/libsvm-model,/'
  awk 'NR==1{a=$1} NR==2{b=$1} END{if(NR!=2 || a>=-0.8 || b<=0.8 || a<=-1.2 || b>=1.2) exit 1; printf "libsvm-probe,cli,%s,%s,pass\n",a,b}' "$dir/pred"
else
  echo "libsvm-probe,cli,unavailable"
fi
