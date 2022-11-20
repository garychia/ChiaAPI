#include "Tester.hpp"

namespace ChiaTest
{
Tester::Tester() : tests()
{
}

Tester::~Tester()
{
    for (size_t i = 0; i < tests.Length(); i++)
        delete tests[i];
}
} // namespace ChiaTest
