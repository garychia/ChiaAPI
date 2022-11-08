#include "Exceptions.hpp"

namespace ChiaRuntime
{

Exception::Exception(const String &msg) : msg(msg)
{
}

String Exception::GetMessage() const noexcept
{
    return msg;
}

IndexOutOfBound::IndexOutOfBound(const String &msg) : Exception(msg)
{
}

DividedByZero::DividedByZero(const String &msg) : Exception(msg)
{
}

InvalidArgument::InvalidArgument(const String &msg) : Exception(msg)
{
}

NoImplementation::NoImplementation(const String &msg) : Exception(msg)
{
}
} // namespace ChiaRuntime
