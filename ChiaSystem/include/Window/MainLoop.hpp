#ifndef MAIN_LOOP_HPP
#define MAIN_LOOP_HPP

namespace ChiaSystem
{
namespace Window
{
/**
 * @brief Represents a loop that updates resources constantly.
 */
class MainLoop
{
  private:
    /**
     * @brief indicates whether the MainLoop should continue to operate.
     */
    bool shouldContinue;

  public:
    /**
     * @brief Construct a new MainLoop object
     */
    MainLoop();

    /**
     * @brief Destroy the MainLoop object
     */
    virtual ~MainLoop();

    /**
     * @brief Check if the MainLoop should continue to execute
     *
     * @return true if it should continue.
     * @return false otherwise.
     */
    virtual bool ShouldContinue() const;

    /**
     * @brief Update the resources assocated with the MainLoop
     */
    virtual void Execute();
};
} // namespace Window
} // namespace ChiaSystem
#endif // MAIN_LOOP_HPP
