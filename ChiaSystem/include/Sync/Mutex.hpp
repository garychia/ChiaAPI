#ifndef MUTEX_HPP
#define MUTEX_HPP

namespace ChiaSystem
{
namespace Sync
{
class Mutex
{
  private:
    void *handle;

  public:
    Mutex();

    Mutex(const Mutex&) = delete;

    Mutex &operator=(const Mutex &) = delete;

    ~Mutex();

    bool IsValid() const;

    bool Lock();

    bool TryLock();

    bool Unlock();
};
} // namespace Sync
} // namespace ChiaSystem

#endif
