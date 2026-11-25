#include "gemv_tab.h"

extern "C" void inkling_iq4_prep(const bfloat16* __restrict input, uint8_t* __restrict table, int32_t width) {
    gemv_q4_prep(input, table, width);
}
