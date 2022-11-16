#ifndef CHIA_WINDOW_H
#define CHIA_WINDOW_H

#include "App/ChiaApp.hpp"
#include "Str.hpp"
#include "Window/MainLoop.hpp"

namespace ChiaSystem
{
namespace Window
{
struct ChiaWindowInfo
{
    size_t width;
    size_t height;
    size_t x;
    size_t y;
    ChiaData::Str<char> title;
    bool fullScreen;
    bool shown;

    ChiaWindowInfo();
};

class ChiaWindow
{
  protected:
    void *handle;
    ChiaWindowInfo info;
    MainLoop loop;

  public:
    ChiaWindow(const ChiaWindowInfo &info);

    virtual bool Create(App::ChiaApp &app);

    virtual void Update();

    void *GetHandle();

    bool IsRunning() const;

    virtual void OnResize(size_t newWidth, size_t newHeight);

    virtual void OnClose();

    virtual void OnDestroy();
};
} // namespace Window
} // namespace ChiaSystem
#endif // WINDOW_H
