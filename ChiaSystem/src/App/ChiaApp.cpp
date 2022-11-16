#include "App/ChiaApp.hpp"

#ifdef CHIA_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif // CHIA_WINDOWS

namespace ChiaSystem
{
namespace App
{
ChiaApp::ChiaApp(const ChiaAppCreateInfo &info) : name(info.appName), version(version)
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
    wndClass.lpfnWndProc = DefWindowProc;
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

const ChiaApp::AppStrType &ChiaApp::GetName() const
{
    return name;
}
} // namespace App
} // namespace ChiaSystem
