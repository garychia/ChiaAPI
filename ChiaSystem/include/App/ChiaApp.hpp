#ifndef Chia_APP_HPP
#define Chia_APP_HPP

#include "App/MainLoop.hpp"
#include "Str.hpp"
#include "Version.hpp"

namespace ChiaSystem
{
namespace App
{
struct ChiaAppCreateInfo
{
    ChiaData::Str<char> appName;

    Version appVersion;
};

class ChiaApp
{
    using AppStrType = ChiaData::Str<char>;

  protected:
    AppStrType name;

    Version version;

  public:
    ChiaApp(const ChiaAppCreateInfo &info);

    virtual ~ChiaApp();

    virtual bool Initialize();

    virtual void Finalize();

    virtual int Execute() = 0;

    const AppStrType &GetName() const;
};
} // namespace App
} // namespace ChiaSystem
#endif // APP_HPP
