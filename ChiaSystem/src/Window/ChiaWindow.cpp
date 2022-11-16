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

ChiaWindow::ChiaWindow(const ChiaWindowInfo &info) : handle(nullptr), info(info)
{
}

bool ChiaWindow::Create(App::ChiaApp &app)
{
    if (handle)
        return false;
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

    handle = CreateWindowEx(WS_EX_APPWINDOW, app.GetName().CStr(), info.title.CStr(), WS_OVERLAPPEDWINDOW, info.x,
                            info.y, windowWidth, windowHeight, NULL, NULL, GetModuleHandle(NULL), NULL);
    if (!handle)
    {
        if (info.fullScreen)
            ChangeDisplaySettings(NULL, 0);
        return false;
    }
    if (info.shown)
    {
        ShowWindow((HWND)handle, SW_SHOW);
        SetForegroundWindow((HWND)handle);
        SetFocus((HWND)handle);
    }
#endif // CHIA_WINDOWS
    app.RegisterWindow(*this);
    return true;
}

void ChiaWindow::Update()
{
    loop.Execute();
}

void *ChiaWindow::GetHandle()
{
    return handle;
}

bool ChiaWindow::IsRunning() const
{
    return loop.ShouldContinue();
}

void ChiaWindow::OnResize(size_t newWidth, size_t newHeight)
{
    info.width = newWidth;
    info.height = newHeight;
}

void ChiaWindow::OnClose()
{
#ifdef CHIA_WINDOWS
    PostQuitMessage(0);
#endif // CHIA_WINDOWS
}

void ChiaWindow::OnDestroy()
{
#ifdef CHIA_WINDOWS
    if (handle)
        DestroyWindow((HWND)handle);
    handle = nullptr;
#endif // CHIA_WINDOWS
}
} // namespace Window
} // namespace ChiaSystem
