// The pb[5] rows past the last whole block, from the padded window element, one at a time as attn_step takes them.
// With ATTN_KVS_BLOCK_ONLY they are ONE block instead: its real rows first, the slots past them masked.
#include "attn.h"
#if ATTN_KVS
extern "C" {
void attn_tailw(const bfloat16 *__restrict e, const ATTN_QT *__restrict qs, float *__restrict oacc,
                float *__restrict ml, int32_t *__restrict pb ATTN_H0_PARM) {
  constexpr unsigned kS = ATTN_KVS * kHD;
#if ATTN_KVS_BLOCK_ONLY
  // attn_step_impl's mask, for the element at once: rows t >= pos (the position-0 dummy) are not real
  const int32_t left = pb[0] - pb[2];
  const unsigned n = left <= 0 ? 0u : ((unsigned)left < (unsigned)pb[5] ? (unsigned)left : (unsigned)pb[5]);
  pb[2] += pb[5];
  if (n == 0) return;
  const bfloat16 *K[kRB], *V[kRB];
  for (unsigned r = 0; r < kRB; ++r) {
    const unsigned rr = r < n ? r : 0u;          // a masked slot reads row 0, which is real
    K[r] = e + 2 * rr * kS;
    V[r] = e + (2 * rr + 1) * kS;
  }
#if !ATTN_NULL
  attn_rowb_impl(K, V, qs, oacc, ml, n ATTN_H0_ARG);
#else
  (void)qs; (void)oacc; (void)ml;
#endif
#else
  const unsigned n = (unsigned)pb[5];
  for (unsigned r = 0; r < n; ++r) attn_step_impl(e + 2 * r * kS, e + (2 * r + 1) * kS, qs, oacc, ml, pb ATTN_H0_ARG);
#endif
}
}
#endif
