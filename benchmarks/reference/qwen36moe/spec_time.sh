#!/usr/bin/env bash
# Decode ms/token from the wall time of phlegm run at two lengths, speculative off and lookup (2026-11-25).
P=/c/code/phlegm-qwen36moe/target/release/phlegm.exe
O=/c/models/qwen36moe/mtp
PROMPT=${PROMPT:-'Here is a Python class:

class Inventory:
    def __init__(self):
        self.items = {}

    def add_item(self, name, count):
        if name in self.items:
            self.items[name] += count
        else:
            self.items[name] = count

    def remove_item(self, name, count):
        if name in self.items:
            self.items[name] -= count
            if self.items[name] <= 0:
                del self.items[name]

    def total(self):
        return sum(self.items.values())

Rewrite the whole class exactly as it is, but rename Inventory to Stock and items to goods. Output only the code.'}
for mode in ${MODES:-off lookup mtp}; do for n in 8 ${N:-200}; do
  s=$(date +%s%N)
  OFLM_OPEN_MTP=1 OFLM_OPEN_VERIFY=1 "$P" run --model-dir 'C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S' --kernels 'C:\models\qwen36moe\kernels-v' \
    --bridge 'C:\models\qwen36moe\bridge\phlegm-reference-bench.exe' --max-ctx 4096 --think false --temperature 0 \
    --speculative $mode --max-tokens $n --prompt "$PROMPT" > $O/st_${mode}_$n.out 2> $O/st_${mode}_$n.err
  rc=$?; e=$(date +%s%N); echo "$mode $n exit $rc wall_ms $(( (e - s) / 1000000 )) chars $(wc -c < $O/st_${mode}_$n.out)"
done; done
cmp -s $O/st_off_${N:-200}.out $O/st_lookup_${N:-200}.out && echo "OUTPUT IDENTICAL" || echo "OUTPUT DIFFERS"
for m in lookup mtp; do cmp -s $O/st_off_${N:-200}.out $O/st_${m}_${N:-200}.out && echo "$m OUTPUT IDENTICAL" || echo "$m OUTPUT DIFFERS"; done
