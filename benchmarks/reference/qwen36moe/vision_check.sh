#!/usr/bin/env bash
# An image prompt through the CLI on the block route and one step a row, answers and last-position logits compared
# (OPEN-VISION-BLOCK, 2026-11-26): vision_check.sh <kernels> <ids-file> <image.ppm> [tag]
. "$(dirname "$0")/paths.sh"
kern=$1; ids=$2; img=$3; tag=${4:-$(basename "$img" .ppm)}
O=$QW/vision/runs; mkdir -p $O
for route in block seq; do
  extra=(); [ $route = seq ] && extra=(--image-seq)
  OFLM_VIT_PROFILE=1 NPUQ_ETA=${NPUQ_ETA:-15m} "$QD/npuq.sh" "$QCLI" --model "$(qpath "$QM")" --kernels "$kern" \
    --ids-file "$(qpath "$ids")" --image "$(qpath "$img")" "${extra[@]}" --max-tokens ${N:-48} --max-ctx 4096 \
    --dump-logits "$(qpath "$O/${tag}_$route")" > $O/${tag}_$route.log 2>&1
  echo "$route exit $? $(grep -E '^image prompt|^vit:' $O/${tag}_$route.log | tr '\n' ' ')"
  "$QPY" -P -c "
import re, sys
from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[1] + '/tokenizer.json')
ids = [int(x) for x in re.findall(r'^token (\d+)', open(sys.argv[2], errors='replace').read(), re.M)]
print(repr(t.decode(ids)))" "$QM" $O/${tag}_$route.log
done
"$QPY" -P -c "
import glob, sys
import numpy as np
a, b = (np.fromfile(sorted(glob.glob(sys.argv[1] + '_' + r + '_p*.bin'))[-1], '<f4').astype(np.float64) for r in ('block', 'seq'))
pa, pb = (np.exp(x - x.max()) / np.exp(x - x.max()).sum() for x in (a, b))
print('block vs seq, last position: KL %.5f, same top %s, top5 %s / %s' % ((pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30))).sum(),
      a.argmax() == b.argmax(), list(np.argsort(-a)[:5]), list(np.argsort(-b)[:5])))" "$O/$tag"
