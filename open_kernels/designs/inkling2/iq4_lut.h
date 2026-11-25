#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

struct Iq4Decode {
    using Layout = aie::lut<4, int8_t>;
    using Lookup = aie::parallel_lookup<uint8_t, Layout, aie::lut_oor_policy::truncate>;
    mutable Lookup lookup;

    Iq4Decode(const int16_t* bank_a, const int16_t* bank_b) : lookup(Layout(16, bank_a, bank_b)) {}

    aie::vector<int8_t, 64> operator()(const uint8_t* packed) const {
        const auto nibbles = aie::load_v<32>(packed).cast_to<uint4>().unpack();
        const auto lo = lookup.fetch(nibbles.extract<32>(0));
        const auto hi = lookup.fetch(nibbles.extract<32>(1));
        return aie::concat(lo, hi);
    }
};
