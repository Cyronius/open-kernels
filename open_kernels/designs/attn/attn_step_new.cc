// The step for the NEW position, whose k'/v' live in bf16 scratch: never masked
// (a second symbol only because IRON type-checks memref args per ExternalFunction).
// With ATTN_KVS_BLOCK_ONLY it is a block of its own: the new row in slot 0, every other slot masked
// and pointing at it too (attn_stepb_nr's form), so the single-row kernel is not built.
#include "attn.h"
extern "C" {
void attn_step_new(const bfloat16 *__restrict Kt, const bfloat16 *__restrict Vt, const ATTN_QT *__restrict qs,
                   float *__restrict oacc, float *__restrict ml ATTN_H0_PARM) {
#if ATTN_KVS && ATTN_KVS_BLOCK_ONLY
  const bfloat16 *K[kRB], *V[kRB];
  AIE_LOOP_UNROLL_FULL
  for (unsigned r = 0; r < kRB; ++r) {
    K[r] = Kt + attn_kvb(h0) * kHD;
    V[r] = Vt + attn_kvb(h0) * kHD;
  }
#if !ATTN_NULL
  attn_rowb_impl(K, V, qs, oacc, ml, 1u ATTN_H0_ARG);
#else
  (void)K; (void)V; (void)qs; (void)oacc; (void)ml;
#endif
#elif ATTN_KVS
  // attn_row_impl indexes this core's window slice; the scratch row is whole, so start it at the slice
  attn_row_impl(Kt + attn_kvb(h0) * kHD, Vt + attn_kvb(h0) * kHD, qs, oacc, ml ATTN_H0_ARG);
#else
  attn_row_impl(Kt, Vt, qs, oacc, ml ATTN_H0_ARG);
#endif
}
}
