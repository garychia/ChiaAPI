#ifndef TESTER_HPP
#define TESTER_HPP

#include "DynamicArray.hpp"
#include "Test.hpp"

namespace ChiaTest
{
class Tester
{
  private:
    ChiaData::DynamicArray<Test *> tests;

  public:
    Tester();

    virtual ~Tester();

    template <class TestType, class... Args> void AddTest(Args &...args)
    {
        tests.Append(new TestType(args...));
    }
};
} // namespace ChiaTest
#endif // TESTER_HPP
