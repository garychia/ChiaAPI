#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include "String.hpp"

using namespace ChiaData;

namespace ChiaRuntime
{
/**
 * @brief Exception is a general-purpose exception.
 */
class Exception
{
  protected:
    // an message associated with this exception.
    String msg;

  public:
    /**
     * @brief Construct a new Exception object
     *
     * @param msg a message to be associated with this exception.
     */
    Exception(const String &msg = "");

    /**
     * @brief Get the message associated with this exception
     *
     * @return String the message associated with this exception.
     */
    virtual String GetMessage() const noexcept;
};

/**
 * @brief IndexOutOfBound is an exception thrown when an index is out of bound.
 */
class IndexOutOfBound : public Exception
{
  public:
    /**
     * @brief Construct a new IndexOutOfBound object
     *
     * @param msg a message to be associated with this exception.
     */
    IndexOutOfBound(const String &msg = "");
};

/**
 * @brief DividedByZero is an exception thown when a value is being divided by zero.
 */
class DividedByZero : public Exception
{
  public:
    /**
     * @brief Construct a new DividedByZero object
     *
     * @param msg a message to be associated with this exception.
     */
    DividedByZero(const String &msg = "");
};

/**
 * @brief InvalidArgument is an exception thrown when an argument is not valid.
 */
class InvalidArgument : public Exception
{
  public:
    /**
     * @brief Construct a new Invalid Argument object
     *
     * @param msg a message to be associated with this exception.
     */
    InvalidArgument(const String &msg = "");
};

/**
 * @brief NoImplementation is an exception thrown when implementation is not present.
 */
class NoImplementation : public Exception
{
  public:
    /**
     * @brief Construct a new NoImplementation object
     *
     * @param msg a message to be associated with this exception.
     */
    NoImplementation(const String &msg = "");
};
} // namespace ChiaRuntime

#endif
