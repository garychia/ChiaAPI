#ifndef TESTER_HPP
#define TESTER_HPP

#include "DynamicArray.hpp"
#include "Test.hpp"

namespace ChiaTest
{
/**
 * @brief Tester performs multiple Tests to detect bugs.
 */
class Tester
{
  private:
    /**
     * @brief tests to be performed by the Tester
     */
    ChiaData::DynamicArray<Test *> tests;

  public:
    /**
     * @brief Construct a new Tester object
     */
    Tester();

    /**
     * @brief Destroy the Tester object
     */
    virtual ~Tester();

    /**
     * @brief Add a test to the Tester
     *
     * @tparam TestType the type of the test.
     * @tparam Args the types of arguments to be passed to the constructor of test.
     * @param args the arguments to be passed to the constructor of test.
     */
    template <class TestType, class... Args> void AddTest(Args &...args)
    {
        tests.Append(new TestType(args...));
    }
};
} // namespace ChiaTest
#endif // TESTER_HPP
