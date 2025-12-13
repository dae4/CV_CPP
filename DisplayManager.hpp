#pragma once
#include "Common.hpp"

namespace DisplayManager {
    void onMouse(int event, int x, int y, int flags, void* userdata);
    void render(const RuntimeState& state, const AppConfig& config);
    void handleInput(RuntimeState& state);
}