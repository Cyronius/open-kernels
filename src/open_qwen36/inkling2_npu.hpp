#pragma once

#include "open_qwen36/inkling2_host.hpp"

namespace open_qwen36 {

// Load a routed expert bundle as produced by recipes/inkling2_routed.py.
std::unique_ptr<Inkling2ExpertProjector> make_inkling2_npu_experts(const std::string& bundle,
                                                               unsigned timeout_ms = 12000);

}  // namespace open_qwen36
