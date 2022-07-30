#pragma once
#include "../SDK/GameEvent.h"
namespace Dump
{
    void DumpGameData() noexcept;
    void DumpEvent(GameEvent* event) noexcept;
}
