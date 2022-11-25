#include "App/ChiaApp.hpp"
#include "Window/ChiaWindow.hpp"

#ifdef CHIA_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif // CHIA_WINDOWS

namespace ChiaSystem
{
namespace App
{
#ifdef CHIA_WINDOWS
LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_DESTROY: {
        ChiaApp::HandleWindowOnDestroy(hwnd);
        return 0;
    }
    case WM_CLOSE: {
        ChiaApp::HandleWindowOnClose(hwnd);
        return 0;
    }
    case WM_SIZE: {
        RECT rect;
        GetClientRect(hwnd, &rect);
        ChiaApp::HandleWindowOnResize(hwnd, rect.right, rect.bottom);
        return 0;
    }
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}
#endif // CHIA_WINDOWS

ChiaAppCreateInfo::ChiaAppCreateInfo(const AppStrType &name, const Version &version)
    : appName(name), appVersion(version)
{
}

WindowAppMapType ChiaApp::windowHandleAppMap = WindowAppMapType();

void ChiaApp::HandleWindowOnResize(void *handle, size_t newWidth, size_t newHeight)
{
    if (!windowHandleAppMap.Contains(handle))
        return;
    ChiaApp *pApp = windowHandleAppMap[handle];
    pApp->handleWindowMap[handle]->OnResize(newWidth, newHeight);
}

void ChiaApp::HandleWindowOnClose(void *handle)
{
    if (!windowHandleAppMap.Contains(handle))
        return;
    ChiaApp *pApp = windowHandleAppMap[handle];
    pApp->handleWindowMap[handle]->OnClose();
}

void ChiaApp::HandleWindowOnDestroy(void *handle)
{
    if (!windowHandleAppMap.Contains(handle))
        return;
    ChiaApp *pApp = windowHandleAppMap[handle];
    pApp->handleWindowMap[handle]->OnDestroy();
}

ChiaApp::ChiaApp(const ChiaAppCreateInfo &info) : name(info.appName), version(1, 0, 0), handleWindowMap()
{
}

ChiaApp::~ChiaApp()
{
}

bool ChiaApp::Initialize()
{
#ifdef CHIA_WINDOWS
    WNDCLASSEX wndClass = {};
    wndClass.hInstance = GetModuleHandle(NULL);
    wndClass.lpfnWndProc = WndProc;
    wndClass.hIcon = LoadIcon(NULL, IDI_WINLOGO);
    wndClass.hIconSm = wndClass.hIcon;
    wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndClass.lpszClassName = name.CStr();
    wndClass.cbSize = sizeof(WNDCLASSEX);
    return RegisterClassEx(&wndClass);
#endif // CHIA_WINDOWS
}

void ChiaApp::Finalize()
{
#ifdef CHIA_WINDOWS
    UnregisterClass(name.CStr(), GetModuleHandle(NULL));
#endif // CHIA_WINDOWS
}

const AppStrType &ChiaApp::GetName() const
{
    return name;
}

void ChiaApp::RegisterWindow(Window::ChiaWindow &window)
{
    if (handleWindowMap.Contains(window.GetHandle()))
        return;
    handleWindowMap[window.GetHandle()] = &window;
    windowHandleAppMap[window.GetHandle()] = this;
}
} // namespace App
} // namespace ChiaSystem
