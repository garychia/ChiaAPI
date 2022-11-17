#ifndef VERSION_HPP
#define VERSION_HPP

namespace ChiaSystem
{
/**
 * @brief Represents the version of an application.
 *
 */
struct Version
{
    /**
     * @brief major of the Version.
     */
    unsigned int major;
    /**
     * @brief minor of the Version.
     */
    unsigned int minor;
    /**
     * @brief patch of the Version.
     */
    unsigned int patch;

    /**
     * @brief Construct a new Version object
     *
     * @param major the major of the Version.
     * @param minor the minor of the Version.
     * @param patch the patch of the Version.
     */
    Version(unsigned int major, unsigned int minor, unsigned int patch);
};
} // namespace ChiaSystem

#endif
