// DisplayManager.hpp
#pragma once
#include "Common.hpp"

namespace DisplayManager {
    void render(const RuntimeState& state, const AppConfig& config);
    void handleInput(RuntimeState& state);
    void onMouse(int event, int x, int y, int flags, void* userdata);
}