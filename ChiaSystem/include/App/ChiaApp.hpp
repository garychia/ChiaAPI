#ifndef Chia_APP_HPP
#define Chia_APP_HPP

#include "HashTable.hpp"
#include "Str.hpp"
#include "Version.hpp"
#include "Window/MainLoop.hpp"

namespace ChiaSystem
{
namespace Window
{
class ChiaWindow;
} // namespace Window

namespace App
{
class ChiaApp;

using AppStrType = ChiaData::Str<char>;
using HandleWindowMapType = ChiaData::HashTable<void *, Window::ChiaWindow *>;
using WindowAppMapType = ChiaData::HashTable<void *, ChiaApp *>;

struct ChiaAppCreateInfo
{
    AppStrType appName;

    Version appVersion;

    ChiaAppCreateInfo(const AppStrType &name = "", const Version &version = Version(1, 0, 0));
};

class ChiaApp
{
  protected:
    static WindowAppMapType windowHandleAppMap;

    AppStrType name;

    Version version;

    HandleWindowMapType handleWindowMap;

  public:
    static void HandleWindowOnResize(void *handle, size_t newWidth, size_t newHeight);

    static void HandleWindowOnClose(void *handle);

    static void HandleWindowOnDestroy(void *handle);

    ChiaApp(const ChiaAppCreateInfo &info);

    virtual ~ChiaApp();

    virtual bool Initialize();

    virtual void Finalize();

    virtual int Execute() = 0;

    const AppStrType &GetName() const;

    virtual void RegisterWindow(Window::ChiaWindow &window);
};
} // namespace App
} // namespace ChiaSystem
#endif // APP_HPP
