#include "App/ChiaApp.hpp"
#include "Window/ChiaWindow.hpp"

class EmptyApp : public ChiaSystem::App::ChiaApp
{
  public:
    EmptyApp(const ChiaSystem::App::ChiaAppCreateInfo &info);

    ~EmptyApp();

    virtual int Execute() override;
};

EmptyApp::EmptyApp(const ChiaSystem::App::ChiaAppCreateInfo &info) : ChiaSystem::App::ChiaApp(info)
{
}

EmptyApp::~EmptyApp()
{
}

int EmptyApp::Execute()
{
    ChiaSystem::Window::ChiaWindowInfo windowInfo;
    windowInfo.title = "Empty Window";
    windowInfo.width = 500;
    windowInfo.height = 250;
    windowInfo.x = 20;
    windowInfo.y = 20;
    windowInfo.fullScreen = false;
    windowInfo.shown = true;
    ChiaSystem::Window::ChiaWindow emptyWindow(windowInfo);
    if (!emptyWindow.Create(*this))
        return -1;
    while (emptyWindow.IsRunning())
    {
        emptyWindow.Update();
    }
    Finalize();
    return 0;
}

int main()
{
    ChiaSystem::App::ChiaAppCreateInfo appInfo;
    appInfo.appName = "Empty Window App";
    appInfo.appVersion = ChiaSystem::Version(1, 0, 0);
    EmptyApp app(appInfo);
    app.Initialize();
    return app.Execute();
}
