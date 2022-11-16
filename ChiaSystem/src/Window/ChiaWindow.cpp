#include "Window/ChiaWindow.hpp"

#ifdef CHIA_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif // CHIA_WINDOWS

namespace ChiaSystem
{
namespace Window
{
ChiaWindowInfo::ChiaWindowInfo() : width(500), height(500), x(0), y(0), title(), fullScreen(false), shown(true)
{
}

ChiaWindow::ChiaWindow(const ChiaWindowInfo &info, const App::ChiaApp &app) : handle(nullptr), info(info), pApp(&app)
{
}

bool ChiaWindow::Create()
{
#ifdef CHIA_WINDOWS
    const auto windowWidth = info.fullScreen ? GetSystemMetrics(SM_CXSCREEN) : info.width;
    const auto windowHeight = info.fullScreen ? GetSystemMetrics(SM_CYSCREEN) : info.height;
    const int winPosX = info.fullScreen ? 0 : info.x;
    const int winPosY = info.fullScreen ? 0 : info.y;

    if (info.fullScreen)
    {
        DEVMODE devMode = {};
        devMode.dmPelsWidth = (DWORD)windowWidth;
        devMode.dmPelsHeight = (DWORD)windowHeight;
        devMode.dmBitsPerPel = 32;
        devMode.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_BITSPERPEL;
        devMode.dmSize = sizeof(devMode);
        ChangeDisplaySettings(&devMode, CDS_FULLSCREEN);
    }

    handle = CreateWindowEx(WS_EX_APPWINDOW, pApp->GetName().CStr(), info.title.CStr(), WS_OVERLAPPEDWINDOW, info.x,
                            info.y, windowWidth, windowHeight, NULL, NULL, GetModuleHandle(NULL), NULL);
    if (!handle)
    {
        if (info.fullScreen)
            ChangeDisplaySettings(NULL, 0);
        return false;
    }
    return true;
#endif // CHIA_WINDOWS
}

void ChiaWindow::Destroy()
{
#ifdef CHIA_WINDOWS
    if (handle)
        DestroyWindow((HWND)handle);
#endif // CHIA_WINDOWS
}
} // namespace Window
} // namespace ChiaSystem
