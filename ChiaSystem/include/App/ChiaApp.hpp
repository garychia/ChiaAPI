#ifndef Chia_APP_HPP
#define Chia_APP_HPP

#include "HashTable.hpp"
#include "Str.hpp"
#include "Sync/Mutex.hpp"
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
/**
 * @brief type of the name of ChiaApp
 */
using AppStrType = ChiaData::Str<char>;
/**
 * @brief type of the data structure that maps window handles to ChiaWindows
 */
using HandleWindowMapType = ChiaData::HashTable<void *, Window::ChiaWindow *>;
/**
 * @brief type of the data structure that maps window handles to ChiaApps.
 */
using WindowAppMapType = ChiaData::HashTable<void *, ChiaApp *>;

/**
 * @brief Contains information about a ChiaApp object to be created.
 */
struct ChiaAppCreateInfo
{
    /**
     * @brief the name of application
     */
    AppStrType appName;

    /**
     * @brief the version of application
     */
    Version appVersion;

    /**
     * @brief Construct a new ChiaAppCreateInfo object
     *
     * @param name the name of ChiaApp object to be created.
     * @param version the version of ChiaApp object to be created.
     */
    ChiaAppCreateInfo(const AppStrType &name = "", const Version &version = Version(1, 0, 0));
};

/**
 * @brief Represents an application that is capable of creating and handling windows.
 */
class ChiaApp
{
  protected:
    static Sync::Mutex windowHandleAppMapMutex;

    /**
     * @brief maps a window handle to the ChiaApp object which created it.
     */
    static WindowAppMapType windowHandleAppMap;

    /**
     * @brief identifies the ChiaApp.
     */
    AppStrType name;

    /**
     * @brief the version of ChiaApp.
     */
    Version version;

    Sync::Mutex handleWindowMapMutex;

    /**
     * @brief maps a window handle to the ChiaWindow which owns it.
     */
    HandleWindowMapType handleWindowMap;

  public:
    /**
     * @brief Handle the window being resized
     *
     * @param handle the handle of the window being resized.
     * @param newWidth the new width of the window being resized.
     * @param newHeight the new height of the window being resize.
     */
    static void HandleWindowOnResize(void *handle, size_t newWidth, size_t newHeight);

    /**
     * @brief Handle the window being closed
     *
     * @param handle the handle of the window being closed.
     */
    static void HandleWindowOnClose(void *handle);

    /**
     * @brief Handle the window being destroyed
     *
     * @param handle the handle of the window being destroyed.
     */
    static void HandleWindowOnDestroy(void *handle);

    /**
     * @brief Construct a new ChiaApp object
     *
     * @param info the information about the new ChiaApp object to be initialized.
     */
    ChiaApp(const ChiaAppCreateInfo &info);

    /**
     * @brief Destroy the Chia App object
     */
    virtual ~ChiaApp();

    /**
     * @brief Initialize the ChiaApp
     *
     * @return true if the initialization was done successfully.
     * @return false otherwise.
     */
    virtual bool Initialize();

    /**
     * @brief Finalize and release resources associated with the ChiaApp
     */
    virtual void Finalize();

    /**
     * @brief Execute the ChiaApp
     *
     * @return int the status code representing if this ChiaApp executed successfully or not. (0 stands for success)
     */
    virtual int Execute() = 0;

    /**
     * @brief Get the Name of the ChiaApp
     *
     * @return const AppStrType& the name.
     */
    const AppStrType &GetName() const;

    /**
     * @brief Register a ChiaWindow to be created and handled by the ChiaApp.
     *
     * @param window the ChiaWindow to be associated with this ChiaApp.
     */
    virtual void RegisterWindow(Window::ChiaWindow &window);
};
} // namespace App
} // namespace ChiaSystem
#endif // APP_HPP
