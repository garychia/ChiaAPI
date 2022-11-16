#include "App/MainLoop.hpp"

namespace ChiaSystem
{
namespace App
{
MainLoop::MainLoop() : shouldContinue(true)
{
}

MainLoop::~MainLoop()
{
}

bool MainLoop::ShouldContinue() const
{
    return shouldContinue;
}

} // namespace App
} // namespace ChiaSystem
