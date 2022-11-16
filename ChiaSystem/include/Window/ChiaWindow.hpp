#ifndef CHIA_WINDOW_H
#define CHIA_WINDOW_H

#include "App/ChiaApp.hpp"
#include "Str.hpp"

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
    const App::ChiaApp *pApp;

  public:
    ChiaWindow(const ChiaWindowInfo &info, const App::ChiaApp &app);

    virtual bool Create();

    virtual void Destroy();
};
} // namespace Window
} // namespace ChiaSystem
#endif // WINDOW_H
