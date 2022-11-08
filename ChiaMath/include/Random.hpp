#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "DynamicArray.hpp"

namespace ChiaMath
{
class Random
{
  private:
    // the random seed.
    static int seed;

    // whether the seed is used.
    static bool useSeed;

    /**
     * @brief Generate a random value.
     *
     * @return int the random value.
     */
    static int Generate();

  public:
    /**
     * @brief Set the Seed value to a new value
     *
     * @param randomSeed the new random seed.
     */
    static void SetSeed(int randomSeed);

    /**
     * @brief Remove the random seed
     */
    static void UnsetSeed();

    /**
     * @brief Generate a random integer
     *
     * @param low the minimum value of the output (inclusive).
     * @param high the maximum value of the output (exclusive).
     * @return int a random integer with the given range.
     */
    static int IntRange(int low, int high);

    /**
     * @brief Choose an integer based on a probability distribution
     *
     * @param nElements the total number of elements that can be chosen.
     * @param prob the probability distribution that decides the probability of each element.
     * @return size_t the index of the element selected. (0 stands for the first one).
     */
    static size_t Choose(size_t nElements,
                         const ChiaData::DynamicArray<double> &prob = ChiaData::DynamicArray<double>());

    /**
     * @brief Generate a random number sampled from a normal distribution
     *
     * @param mean the mean of the normal distribution.
     * @param standard the standard deviation of the distribution.
     * @param nSamples the total number of samples from the distribution to choose from.
     * @return double a random number sampled from a normal distribution.
     */
    static double NormalDistribution(double mean, double standard, unsigned int nSamples = 1000);
};
} // namespace ChiaMath

#endif // RANDOM_HPP
