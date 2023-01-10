#ifndef DIRECTX_HELPER_HPP
#define DIRECTX_HELPER_HPP

#include <d3d11_1.h>
#include <wrl\client.h>

using namespace Microsoft::WRL;

namespace ChiaSystem
{
namespace DX
{
class DXHelper
{
  public:
    static HRESULT CreateDevice(ComPtr<ID3D11Device> &pDevice, ComPtr<ID3D11DeviceContext> &pContext,
                                D3D_FEATURE_LEVEL &featureLevel);

    static HRESULT CreateSwapChain(HWND windowHandle, bool fullScreen, ComPtr<IDXGISwapChain> &pSwapChain,
                                   ComPtr<ID3D11Device> &pDevice, UINT nBuffers = 2);

    static HRESULT CreateRenderTarget(ComPtr<ID3D11Device> &pDevice, ComPtr<IDXGISwapChain> &pSwapChain,
                                      ComPtr<ID3D11Texture2D> &pBackBuffer,
                                      ComPtr<ID3D11RenderTargetView> &pRenderTarget,
                                      D3D11_TEXTURE2D_DESC &backBufferDesc);

    static HRESULT CreateDepthStencilBuffer(ComPtr<ID3D11Device> &pDevice, CD3D11_TEXTURE2D_DESC &desc,
                                            ComPtr<ID3D11Texture2D> &pDepthStencil,
                                            ComPtr<ID3D11DepthStencilView> &pDepthStencilView);

    void CreateViewport(D3D11_VIEWPORT &viewport, D3D11_TEXTURE2D_DESC &backBufferDesc,
                        ComPtr<ID3D11DeviceContext> &pDeviceContext);

    static HRESULT LoadVertexShader(ComPtr<ID3D11Device> &pDevice, LPVOID byteCode, size_t codeSize,
                                    ComPtr<ID3D11VertexShader> &pVertexShader,
                                    const D3D11_INPUT_ELEMENT_DESC *inputDescs, size_t nInputDescs,
                                    ComPtr<ID3D11InputLayout> &pInputLayout);

    static HRESULT LoadPixelShader(ComPtr<ID3D11Device> &pDevice, LPVOID byteCode, size_t codeSize,
                                   ComPtr<ID3D11PixelShader> &pPixelShader);
};
} // namespace DX
} // namespace ChiaSystem
#endif
