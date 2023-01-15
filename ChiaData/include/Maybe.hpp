#ifndef MAYBE_HPP
#define MAYBE_HPP

namespace ChiaData
{
/**
 * @brief A data structure that may or may not contain a value.
 *
 * @tparam T the type of the value.
 */
template <class T> class Maybe
{
  private:
    /**
     * @brief whether the Maybe contains any value.
     */
    bool valid;

    /**
     * @brief the value stored.
     */
    T data;

  public:
    /**
     * @brief Construct a new Maybe object that contains no value.
     */
    Maybe() : valid(false), data()
    {
    }

    /**
     * @brief Construct a new Maybe object that contains a value.
     *
     * @tparam U the type of the value.
     * @param value the value to be stored in the Maybe.
     */
    template <class U> Maybe(const U &value) : valid(true), data(value)
    {
    }

    /**
     * @brief Check if the Maybe contains any value.
     *
     * @return true if it contains any value.
     * @return false otherwise.
     */
    bool IsValid() const
    {
        return valid;
    }

    /**
     * @brief Remove the stored value in the Maybe.
     */
    void Remove()
    {
        valid = false;
    }

    /**
     * @brief Get the value out of the Maybe without checking.
     *
     * @return T& the value.
     */
    T &Get()
    {
        return data;
    }

    /**
     * @brief Get the value out of the Maybe without checking.
     *
     * @return const T& the value.
     */
    const T &Get() const
    {
        return data;
    }

    /**
     * @brief Convert the Maybe into a bool.
     *
     * @return true if the Maybe contains any value.
     * @return false otherwise.
     */
    operator bool() const
    {
        return valid;
    }

    /**
     * @brief Store a given value.
     *
     * @tparam U the type of the value.
     * @param value the value to be stored.
     * @return Maybe<T>& the Maybe.
     */
    template <class U> Maybe<T> &operator=(const U &value)
    {
        valid = true;
        data = value;
        return *this;
    }

    /**
     * @brief Maybe Copy Assignment
     *
     * @tparam U the type of value of the other Maybe.
     * @param other the other Maybe to be copied.
     * @return Maybe<T>& this Maybe.
     */
    template <class U> Maybe<T> &operator=(const Maybe<U> &other)
    {
        valid = other.valid;
        data = other.data;
        return *this;
    }

    /**
     * @brief Check if two Maybes are valid and contain the same value.
     *
     * @tparam U the type of value of the other Maybe.
     * @param other the other Maybe.
     * @return true if both are valid and contain the same value.
     * @return false otherwise.
     */
    template <class U> bool operator==(const Maybe<U> &other) const
    {
        return valid && other.valid && data == other.data;
    }

    /**
     * @brief Check if any of two Maybes is invalid or they contain different values.
     *
     * @tparam U the type of value of the other Maybe.
     * @param other the other Maybe.
     * @return true if any of the Maybes is invalid or they contain different values.
     * @return false otherwise.
     */
    template <class U> bool operator!=(const Maybe<U> &other) const
    {
        return !operator==(other);
    }

    template <class U> friend class Maybe;
};
} // namespace ChiaData

#endif
