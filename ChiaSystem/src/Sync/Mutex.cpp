#include "Sync/Mutex.hpp"

#ifdef CHIA_WINDOWS
#else
#include <pthread.h>
#endif

namespace ChiaSystem
{
namespace Sync
{
Mutex::Mutex() : handle(nullptr)
{
#ifdef CHIA_WINDOWS
#else
    handle = new pthread_mutex_t;
    const auto result = pthread_mutex_init((pthread_mutex_t *)handle, NULL);
    if (result != 0)
    {
        delete (pthread_mutex_t *)handle;
        handle = nullptr;
    }
#endif
}

Mutex::~Mutex()
{

    if (handle)
    {
#ifdef CHIA_WINDOWS
#else
        pthread_mutex_destroy((pthread_mutex_t *)handle);
        delete (pthread_mutex_t *)handle;
#endif
    }
}

bool Mutex::IsValid() const
{
    return !!handle;
}

bool Mutex::Lock()
{
#ifdef CHIA_WINDOWS
#else
    const auto result = pthread_mutex_lock((pthread_mutex_t *)handle);
    return result == 0;
#endif
}

bool Mutex::TryLock()
{
#ifdef CHIA_WINDOWS
#else
    const auto result = pthread_mutex_trylock((pthread_mutex_t *)handle);
    return result == 0;
#endif
}

bool Mutex::Unlock()
{
#ifdef CHIA_WINDOWS
#else
    const auto result = pthread_mutex_unlock((pthread_mutex_t *)handle);
    return result == 0;
#endif
}
} // namespace Sync
} // namespace ChiaSystem
