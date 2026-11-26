// Grouped expert projections plus core SiTU for 16 token rows, sharing SiTU with the singleton image.
#include "block_expert.cc"
#include "situ.h"

// Three K32 down-input tiles (1,280 B each) from one gate band and its up band, all 16 rows, into `frame`.
extern "C" void inkling_grouped_situ(const float* gate, const float* up, float* frame) {
    aie::set_rounding(aie::rounding_mode::conv_even);
    auto* wire = reinterpret_cast<int16_t*>(frame);
#pragma clang loop unroll(disable)
    for (unsigned t = 0; t < 3; ++t) {
        int16_t* tile = wire + 640 * t;
#pragma clang loop unroll(disable)
        for (unsigned r = 0; r < 16; ++r) {
            const unsigned at = (r / 4) * 384 + 128 * t + (r % 4) * 8;
            int16_t scale;
            const auto codes = inkling_situ::row(gate + at, up + at, scale);
            aie::store_v(tile + 8 * r, aie::broadcast<int16_t, 8>(scale));
#pragma clang loop unroll(full)
            for (unsigned c = 0; c < 4; ++c)
                aie::store_v(tile + 128 + ((r / 4) * 16 + c * 4 + r % 4) * 8, codes.extract<8>(c));
        }
    }
}
