#include "Window/MainLoop.hpp"

#ifdef CHIA_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif // CHIA_WINDOWS

namespace ChiaSystem
{
namespace Window
{
MainLoop::MainLoop() : shouldContinue(true)
{
}

MainLoop::~MainLoop()
{
}

bool MainLoop::ShouldContinue() const
{
    return shouldContinue;
}

void MainLoop::Execute()
{
#ifdef CHIA_WINDOWS
    static MSG msg;
    msg = {};
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    shouldContinue = msg.message != WM_QUIT;
#endif // CHIA_WINDOWS
}

} // namespace Window
} // namespace ChiaSystem
