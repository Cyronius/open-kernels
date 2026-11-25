#ifdef INKLING2_SINGLETON
#include "iq4_lut.h"

// Both expert formats and the dense projections share one worker image.
#define INKLING2_EXPERT_IMPL_ONLY
#define IQ4_WEIGHTS 1
namespace iq4 {
#include "block_expert.cc"
}
#undef IQ4_WEIGHTS
#undef SPLIT_SCALE
#define IQ4_WEIGHTS 0
#define SPLIT_SCALE 1
namespace q8 {
#include "block_expert.cc"
}
#else
#include "block_expert.cc"
#endif

namespace {
constexpr unsigned activation_bytes = 320;
constexpr unsigned activation_group = 4;
#ifdef INKLING2_SINGLETON
constexpr unsigned weight_bytes = 96 * 36;
#else
constexpr unsigned weight_bytes = 96 * (IQ4_WEIGHTS ? 18 : 34);
#endif
constexpr unsigned query_bytes = 9344;
constexpr unsigned result_values = 2080;
constexpr unsigned frame_values = 384;

void copy_bytes(const uint8_t* src, uint8_t* dst, unsigned bytes) {
    for (unsigned i = 0; i < bytes; i += 64) {
        const auto x = aie::load_v<32>(reinterpret_cast<const uint16_t*>(src + i));
        aie::store_v(reinterpret_cast<uint16_t*>(dst + i), x);
    }
}
} // namespace

#ifdef INKLING2_SINGLETON
extern "C" void inkling_shared_q8(const uint8_t* __restrict a, const uint8_t* __restrict w,
                                  const int16_t* lut0, const int16_t* lut1, float* __restrict y,
                                  int32_t within, int32_t first) {
    q8::apply(a + within * activation_bytes, w, lut0, lut1, y, first != 0);
}

extern "C" void inkling_shared_iq4_pair(const uint8_t* __restrict a, const uint8_t* __restrict w,
                                      const int16_t* lut0, const int16_t* lut1, float* __restrict y,
                                      int32_t within, int32_t first) {
    iq4::apply(a + within * activation_bytes, w, lut0, lut1, y, first != 0);
    iq4::apply(a + (within + 1) * activation_bytes, w + weight_bytes / 2, lut0, lut1, y, false);
}
#else
extern "C" void inkling_shared_expert(const uint8_t* __restrict a, const uint8_t* __restrict w,
                                      const int16_t* lut0, const int16_t* lut1, float* __restrict y,
                                      int32_t within, int32_t first) {
    inkling_block_expert(a + within * activation_bytes, w, lut0, lut1, y, first);
}
#endif

extern "C" void inkling_shared_query(const uint8_t* __restrict src, uint8_t* __restrict query, int32_t frame) {
    const unsigned at = frame * weight_bytes;
    const unsigned remaining = query_bytes - at;
    copy_bytes(src, query + at, remaining < weight_bytes ? remaining : weight_bytes);
}

extern "C" void inkling_shared_kv(const uint8_t* __restrict src, uint8_t* __restrict kv, int32_t frame) {
    constexpr unsigned bytes = activation_group * activation_bytes;
    copy_bytes(src, kv + frame * bytes, bytes);
}

extern "C" void inkling_shared_output(const float* __restrict result, float* __restrict out, int32_t frame) {
    const unsigned at = frame * frame_values;
    const unsigned remaining = result_values - at;
    const unsigned copy = remaining < frame_values ? remaining : frame_values;
    for (unsigned i = 0; i < copy; i += 32) aie::store_v(out + i, aie::load_v<32>(result + at + i));
    for (unsigned i = copy; i < frame_values; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}

extern "C" void inkling_shared_zero(float* out) {
    for (unsigned i = 0; i < frame_values; i += 32) aie::store_v(out + i, aie::zeros<float, 32>());
}
