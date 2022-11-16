#ifndef MAIN_LOOP_HPP
#define MAIN_LOOP_HPP

namespace ChiaSystem
{
namespace Window
{
class MainLoop
{
  private:
    bool shouldContinue;

  public:
    MainLoop();

    virtual ~MainLoop();

    virtual bool ShouldContinue() const;

    virtual void Execute();
};
} // namespace Window
} // namespace ChiaSystem
#endif // MAIN_LOOP_HPP
