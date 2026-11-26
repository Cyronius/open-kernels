#!/usr/bin/env bash
# make a test dir for a build: fa256_mk.sh <build> <testdir> <L> [extra make_test args] (2026-11-25)
. "$(dirname "$0")/paths.sh"
b=$1; t=$2; L=$3; shift 3
"$QPY" "$QR/open_kernels/designs/attn_fa256/make_test.py" --build "$QW/fa256/$b" --out "$QW/fa256/$t" --L $L --max-rows $L --runs 5 "$@" | tail -1
