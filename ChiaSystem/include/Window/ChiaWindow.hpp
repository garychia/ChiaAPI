#ifndef CHIA_WINDOW_H
#define CHIA_WINDOW_H

#include "App/ChiaApp.hpp"
#include "Str.hpp"
#include "Window/MainLoop.hpp"

namespace ChiaSystem
{
namespace Window
{
/**
 * @brief Contains information about a ChiaWindow to be created.
 */
struct ChiaWindowInfo
{
    /**
     * @brief width of the ChiaWindow to be created
     */
    size_t width;
    /**
     * @brief height of the ChiaWindow to be created
     */
    size_t height;
    /**
     * @brief horizontal position of the ChiaWindow to be created. (top-left is the origin)
     */
    size_t x;
    /**
     * @brief vertical position of the ChiaWindow to be created. (top-left is the origin)
     */
    size_t y;
    /**
     * @brief title of the ChiaWindow to be created.
     */
    ChiaData::Str<char> title;
    /**
     * @brief indicates whether the ChiaWindow will be in the full-screen mode.
     */
    bool fullScreen;
    /**
     * @brief indicates whether the ChiaWindow will be shown or not.
     */
    bool shown;

    /**
     * @brief Construct a new ChiaWindowInfo object
     */
    ChiaWindowInfo();
};

/**
 * @brief Represents a destop window
 */
class ChiaWindow
{
  protected:
    /**
     * @brief handle of the ChiaWindow.
     */
    void *handle;
    /**
     * @brief information about the ChiaWindow.
     */
    ChiaWindowInfo info;
    /**
     * @brief loop that updates each frame of the ChiaWindow.
     */
    MainLoop loop;

  public:
    /**
     * @brief Construct a new ChiaWindow object
     *
     * @param info the information about the ChiaWindow to be initialized.
     */
    ChiaWindow(const ChiaWindowInfo &info);

    /**
     * @brief Create and display (if set to be shown) the ChiaWindow
     *
     * @param app the ChiaApp the ChiaWindow belongs to.
     * @return true if the creation was successful.
     * @return false otherwise.
     */
    virtual bool Create(App::ChiaApp &app);

    /**
     * @brief Update the frame of the ChiaWindow
     */
    virtual void Update();

    /**
     * @brief Get the Handle object
     *
     * @return void* the handle of the ChiaWindow.
     */
    void *GetHandle();

    /**
     * @brief Check if the ChiaWindow is currently running or not
     *
     * @return true if the ChiaWindow is still running.
     * @return false otherwise.
     */
    bool IsRunning() const;

    /**
     * @brief Update the ChiaWindow when being resized
     *
     * @param newWidth the new width of the ChiaWindow resized.
     * @param newHeight the new height of the ChiaWindow resized.
     */
    virtual void OnResize(size_t newWidth, size_t newHeight);

    /**
     * @brief Update the ChiaWindow when being closed
     */
    virtual void OnClose();

    /**
     * @brief Update the ChiaWindow when being destroyed
     */
    virtual void OnDestroy();
};
} // namespace Window
} // namespace ChiaSystem
#endif // WINDOW_H
