#ifndef PAIR_HPP
#define PAIR_HPP

namespace ChiaData
{
/**
 * @brief A data structure that stores a pair of values, the key and value.
 * 
 * @tparam T the type of the first value (key).
 * @tparam U the type of the second value (value).
 */
template <class T, class U> class Pair
{
  private:
    // the first value of pair.
    T key;

    // the second value of pair.
    U value;

  public:
    /**
     * @brief Retrieve the key of the pair.
     *
     * @return T& the key.
     */
    T &Key()
    {
        return key;
    }

    /**
     * @brief Retrieve the key of the pair.
     *
     * @return const T& the key.
     */
    const T &Key() const
    {
        return key;
    }

    /**
     * @brief Retrieve the value of the pair.
     *
     * @return U& the value.
     */
    U &Value()
    {
        return value;
    }

    /**
     * @brief Retrieve the value of the pair.
     *
     * @return const U& the value.
     */
    const U &Value() const
    {
        return value;
    }

    /**
     * @brief Copy Assignment of Pair
     *
     * @param other the pair to be copied.
     * @return Pair& this pair.
     */
    Pair &operator=(const Pair &other)
    {
        key = other.key;
        value = other.value;
        return *this;
    }
};
} // namespace ChiaData

#endif // PAIR_HPP
