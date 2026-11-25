#pragma once

#include "open_qwen36/vision/inkling2_vision.hpp"

namespace open_qwen36::vision {

std::unique_ptr<Inkling2VisionBackend> make_inkling2_vision_npu(const std::string& bundle, unsigned timeout_ms);

}  // namespace open_qwen36::vision
