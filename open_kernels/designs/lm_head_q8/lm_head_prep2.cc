// Token `tok`'s K-wide bf16 activation into the 2-token table (gptoss/gx_rows.h's layout) for the two-row head.
#include "lm_head_q8.h"
#include "../gptoss/gx_rows.h"

extern "C" {
void lm_head_prep2(const bfloat16 *__restrict x, uint8_t *__restrict tab, int tok) {
  gx_prep2_tok<false>(x, tab, kK, (unsigned)tok, 0, kK / 32);
}
}
