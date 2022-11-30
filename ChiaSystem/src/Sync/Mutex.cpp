#include "Sync/Mutex.hpp"

#ifdef CHIA_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <synchapi.h>
#else
#include <pthread.h>
#endif

namespace ChiaSystem
{
namespace Sync
{
Mutex::Mutex() : handle(nullptr), locked(false)
{
#ifdef CHIA_WINDOWS
    handle = CreateMutexExA(nullptr, nullptr, 0, NULL);
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
        if (locked)
            Unlock();
#ifdef CHIA_WINDOWS
        CloseHandle((HANDLE)handle);
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
    auto result = WaitForSingleObject((HANDLE)handle, INFINITE);
    return result == WAIT_OBJECT_0;
#else
    const auto result = pthread_mutex_lock((pthread_mutex_t *)handle);
    return locked = result == 0;
#endif
}

bool Mutex::TryLock()
{
#ifdef CHIA_WINDOWS
    auto result = WaitForSingleObject((HANDLE)handle, 0);
    return result == WAIT_OBJECT_0;
#else
    const auto result = pthread_mutex_trylock((pthread_mutex_t *)handle);
    return locked = result == 0;
#endif
}

bool Mutex::Unlock()
{
#ifdef CHIA_WINDOWS
    return ReleaseMutex((HANDLE)handle);
#else
    const auto result = pthread_mutex_unlock((pthread_mutex_t *)handle);
    return locked = result != 0;
#endif
}
} // namespace Sync
} // namespace ChiaSystem
