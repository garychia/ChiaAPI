#ifndef MAIN_LOOP_HPP
#define MAIN_LOOP_HPP

namespace ChiaSystem
{
namespace App
{
class MainLoop
{
  private:
    bool shouldContinue;

  public:
    MainLoop();

    virtual ~MainLoop();

    virtual bool ShouldContinue() const;

    virtual void Execute() = 0;
};

} // namespace App
} // namespace ChiaSystem
#endif // MAIN_LOOP_HPP
