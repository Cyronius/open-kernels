# make a test dir for a build: mk.sh <build> <testdir> <L> [extra make_test args] (2026-11-25)
b=$1; t=$2; L=$3; shift 3
python /c/code/open-kernels-qwen36moe/open_kernels/designs/attn_fa256/make_test.py --build /c/models/qwen36moe/fa256/$b --out /c/models/qwen36moe/fa256/$t --L $L --max-rows $L --runs 5 "$@" | tail -1
