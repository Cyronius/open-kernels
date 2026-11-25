#include "gemv_q8.h"

// A 10,240-byte FIFO element holds two Q8 half-tiles or four IQ4 half-tiles.
extern "C" void inkling_q8_group(const uint8_t* __restrict weights, const uint8_t* __restrict tab,
                                float* __restrict output, int32_t group, int32_t width) {
    const unsigned kt = group / 2;
    float* y = output + (group % 2) * 2 * kQ8Rows;
    gemv_q8_half_tile(weights, tab, width, kt, kt == 0, y);
    gemv_q8_half_tile(weights + kQ8TileBytes, tab, width, kt, kt == 0, y + kQ8Rows);
}
