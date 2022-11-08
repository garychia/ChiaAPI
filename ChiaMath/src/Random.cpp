#include "Random.hpp"
#include "Math.hpp"

#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif
namespace ChiaMath
{
int Random::seed = 0;

bool Random::useSeed = false;

void Random::SetSeed(int randomSeed)
{
    seed = randomSeed;
    useSeed = true;
}

void Random::UnsetSeed()
{
    useSeed = false;
}

int Random::Generate()
{
    int randomValue;
#pragma omp critical
    {
#ifdef _OPENMP
        srand((useSeed ? seed : clock()) + omp_get_thread_num());
#else
        srand(useSeed ? seed : clock());
#endif
        randomValue = rand();
    }
    return randomValue;
}

int Random::IntRange(int low, int high)
{
    return Generate() % (high - low) + low;
}

size_t Random::Choose(size_t nElements, const ChiaData::DynamicArray<double> &prob)
{
    if (prob.IsEmpty())
        return IntRange(0, nElements);
    const double randomProb = (double)(Generate() % 1001) / 1000;
    double currentProbability = 0.0;
    for (size_t i = 0; i < prob.Length(); i++)
    {
        currentProbability += prob[i];
        if (randomProb <= currentProbability)
            return i;
    }
    return prob.Length() - 1;
}

double Random::NormalDistribution(double mean, double standard, unsigned int nSamples)
{
    const auto rangeMin = mean - 3 * standard;
    const auto rangeMax = mean + 3 * standard;
    const auto rangeSize = rangeMax - rangeMin;
    const auto stepSize = rangeSize / nSamples;
    ChiaData::DynamicArray<double> distribution;
    ChiaData::DynamicArray<double> values;
    for (auto i = rangeMin; i <= rangeMax; i += stepSize)
    {
        distribution.Append(Gauss(i, mean, standard) * stepSize);
        values.Append(i);
    }
    return values[Choose(distribution.Length(), distribution)];
}
} // namespace ChiaMath
