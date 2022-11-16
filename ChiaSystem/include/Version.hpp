#ifndef VERSION_HPP
#define VERSION_HPP

namespace ChiaSystem
{
struct Version
{
    unsigned int major;
    unsigned int minor;
    unsigned int patch;

    Version(unsigned int major, unsigned int minor, unsigned int patch);
};
} // namespace ChiaSystem

#endif
