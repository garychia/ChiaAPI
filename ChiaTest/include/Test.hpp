#ifndef TEST_HPP
#define TEST_HPP

#include <iostream>

#include "Math.hpp"

#define CHECK_IF_TRUE(expr)                                                                                            \
    if (expr)                                                                                                          \
    {                                                                                                                  \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        std::cout << "In file " << __FILE__ << std::endl;                                                              \
        std::cout << "At line " << __LINE__ << std::endl;                                                              \
        std::cout << "ChiaTest: The condition (" #expr ") is expected to be true but was evaluated to false."          \
                  << std::endl;                                                                                        \
    }

#define CHECK_IF_EQUAL(x, y) CHECK_IF_TRUE((x) == (y))
#define CHECK_IF_NOT_EQUAL(x, y) CHECK_IF_TRUE((x) != (y))
#define CHECK_IF_NEAR(type, x, y) CHECK_IF_TRUE(ChiaMath::Abs((type)((x) - (y)) / (type)(y)) < 0.001)
#define CHECK_IF_NEAR_FLOAT(x, y) CHECK_IF_NEAR(float, x, y)
#define CHECK_IF_NEAR_DOUBLE(x, y) CHECK_IF_NEAR(double, x, y)

namespace ChiaTest
{
class Test
{
  public:
    virtual bool Run() = 0;
};
} // namespace ChiaTest
#endif // TEST_HPP
