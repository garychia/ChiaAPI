#ifndef BINARY_SEARCH_HPP
#define BINARY_SEARCH_HPP

#include "Array.hpp"

namespace ChiaAlgs
{
/**
 * @brief Perform Binary Search on a sorted array-like data structure
 *
 * @tparam IndexType the type of the index.
 * @tparam ArrayLike the type of the array-like data structure.
 * @tparam TargetType the type of the target element.
 * @tparam ArraySize the size of the array-like data structure.
 * @param arr an array-like data structure.
 * @param target the target to search for.
 * @param start the beginning index to be searched. (inclusive) (optional)
 * @param end the last index to be searched. (inclusive) (optional)
 * @return IndexType the index where the target is found. -1 is returned if it does not exist.
 */
template <class IndexType, class ArrayLike, class TargetType, size_t ArraySize>
IndexType BinarySearch(const ArrayLike &arr, const TargetType &target, IndexType start = 0, IndexType end = -1);

/**
 * @brief Search for indices where the target to be searched for occurs in a sorted array-like data structure
 *
 * @tparam IndexType the type of the index.
 * @tparam ArrayLike the type of the array-like data structure.
 * @tparam TargetType the type of the target element.
 * @tparam ArraySize the size of the array-like data structure.
 * @param arr the sorted array-like data structure.
 * @param target the target element to be searched for.
 * @param start the beginning index within the search range. (inclusive) (optional)
 * @param end the last index within the search range. (inclusive) (optional)
 * @return ChiaData::Array<IndexType>
 */
template <class IndexType, class ArrayLike, class TargetType, size_t ArraySize>
ChiaData::Array<IndexType> SearchRange(const ArrayLike &arr, const TargetType &target, IndexType start = 0,
                                       IndexType end = -1);
} // namespace ChiaAlgs

namespace ChiaAlgs
{
template <class IndexType, class ArrayLike, class TargetType, size_t ArraySize>
IndexType BinarySearch(const ArrayLike &arr, const TargetType &target, IndexType start, IndexType end)
{
    if (end == -1)
        end = ArraySize - 1;
    while (start <= end)
    {
        const IndexType middle = start + (end - start) / 2;
        if (arr[middle] == target)
            return middle;
        else if (target < arr[middle])
            end = middle - 1;
        else
            start = middle + 1;
    }
    return -1;
}

template <class IndexType, class ArrayLike, class TargetType, size_t ArraySize>
ChiaData::Array<IndexType> SearchRange(const ArrayLike &arr, const TargetType &target, IndexType start, IndexType end)
{
    IndexType startIndex = -1;
    IndexType endIndex = -1;
    end = (end == -1) ? ArraySize : end + 1;

    while (start < end)
    {
        const IndexType middle = start + (end - start) / 2;
        if (arr[middle] == target)
            end = middle;
        else if (target < arr[middle])
            end = middle;
        else
            start = middle + 1;
    }
    if (start >= ArraySize || arr[start] != target)
        return ChiaData::Array<IndexType>({-1, -1});

    startIndex = start;
    end = ArraySize;
    while (start < end - 1)
    {
        const IndexType middle = start + (end - start) / 2;
        if (arr[middle] == target)
            start = middle;
        else if (target < arr[middle])
            end = middle;
        else
            start = middle + 1;
    }
    endIndex = start;
    if (endIndex >= ArraySize || arr[endIndex] != target)
        return ChiaData::Array<IndexType>({-1, -1});
    return ChiaData::Array<IndexType>({startIndex, endIndex});
}
} // namespace ChiaAlgs

#endif
