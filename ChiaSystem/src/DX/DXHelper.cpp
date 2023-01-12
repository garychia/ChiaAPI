#include "DX/DXHelper.hpp"

#include <dxgi1_3.h>

const D3D_FEATURE_LEVEL DX_SUPPORTED_LEVELS[] = {D3D_FEATURE_LEVEL_11_0};

namespace ChiaSystem
{
namespace DX
{
HRESULT DXHelper::CreateDevice(ComPtr<ID3D11Device> &pDevice, ComPtr<ID3D11DeviceContext> &pContext,
                               D3D_FEATURE_LEVEL &featureLevel)
{
    HRESULT hr;
    UINT deviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#ifndef NDEBUG
    deviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, 0, deviceFlags, DX_SUPPORTED_LEVELS,
                           ARRAYSIZE(DX_SUPPORTED_LEVELS), D3D11_SDK_VERSION, &pDevice, &featureLevel, &pContext);
    return hr;
}

HRESULT DXHelper::CreateSwapChain(HWND windowHandle, bool fullScreen, ComPtr<IDXGISwapChain> &pSwapChain,
                                  ComPtr<ID3D11Device> &pDevice, UINT nBuffers)
{
    HRESULT hr;
    DXGI_SWAP_CHAIN_DESC desc;
    ZeroMemory(&desc, sizeof(DXGI_SWAP_CHAIN_DESC));
    desc.Windowed = fullScreen ? FALSE : TRUE;
    desc.BufferCount = nBuffers;
    desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    desc.OutputWindow = windowHandle;

    ComPtr<IDXGIDevice3> dxgiDevice;
    hr = pDevice.As(&dxgiDevice);
    if (FAILED(hr))
        return hr;

    ComPtr<IDXGIAdapter> pAdapter;
    ComPtr<IDXGIFactory> pFactory;
    hr = dxgiDevice->GetAdapter(&pAdapter);
    if (FAILED(hr))
        return hr;

    pAdapter->GetParent(IID_PPV_ARGS(&pFactory));
    hr = pFactory->CreateSwapChain(pDevice.Get(), &desc, &pSwapChain);
    return hr;
}

HRESULT DXHelper::CreateRenderTarget(ComPtr<ID3D11Device> &pDevice, ComPtr<IDXGISwapChain> &pSwapChain,
                                     ComPtr<ID3D11Texture2D> &pBackBuffer,
                                     ComPtr<ID3D11RenderTargetView> &pRenderTarget,
                                     D3D11_TEXTURE2D_DESC &backBufferDesc)
{
    HRESULT hr = pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **)pBackBuffer.GetAddressOf());
    if (FAILED(hr))
        return hr;
    hr = pDevice->CreateRenderTargetView(pBackBuffer.Get(), nullptr, pRenderTarget.GetAddressOf());
    if (FAILED(hr))
        return hr;
    pBackBuffer->GetDesc(&backBufferDesc);
    return hr;
}

HRESULT DXHelper::CreateDepthStencilBuffer(ComPtr<ID3D11Device> &pDevice, CD3D11_TEXTURE2D_DESC &desc,
                                           ComPtr<ID3D11Texture2D> &pDepthStencil,
                                           ComPtr<ID3D11DepthStencilView> &pDepthStencilView)
{
    HRESULT hr = pDevice->CreateTexture2D(&desc, nullptr, &pDepthStencil);
    if (FAILED(hr))
        return hr;
    CD3D11_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc(D3D11_DSV_DIMENSION_TEXTURE2D);
    hr = pDevice->CreateDepthStencilView(pDepthStencil.Get(), &depthStencilViewDesc, &pDepthStencilView);
    return hr;
}

void DXHelper::CreateViewport(D3D11_VIEWPORT &viewport, D3D11_TEXTURE2D_DESC &backBufferDesc,
                              ComPtr<ID3D11DeviceContext> &pDeviceContext)
{
    ZeroMemory(&viewport, sizeof(D3D11_VIEWPORT));
    viewport.Height = (FLOAT)backBufferDesc.Height;
    viewport.Width = (FLOAT)backBufferDesc.Width;
    viewport.MinDepth = 0;
    viewport.MaxDepth = 1;
    pDeviceContext->RSSetViewports(1, &viewport);
}

HRESULT DXHelper::LoadVertexShader(ComPtr<ID3D11Device> &pDevice, LPVOID byteCode, size_t codeSize,
                                   ComPtr<ID3D11VertexShader> &pVertexShader,
                                   const D3D11_INPUT_ELEMENT_DESC *inputDescs, size_t nInputDescs,
                                   ComPtr<ID3D11InputLayout> &pInputLayout)
{
    HRESULT hr = pDevice->CreateVertexShader(byteCode, codeSize, nullptr, pVertexShader.GetAddressOf());
    if (FAILED(hr))
        return hr;
    hr = pDevice->CreateInputLayout(inputDescs, nInputDescs, byteCode, codeSize, &pInputLayout);
    return hr;
}

HRESULT DXHelper::LoadPixelShader(ComPtr<ID3D11Device> &pDevice, LPVOID byteCode, size_t codeSize,
                                  ComPtr<ID3D11PixelShader> &pPixelShader)
{
    return pDevice->CreatePixelShader(byteCode, codeSize, nullptr, pPixelShader.GetAddressOf());
}
} // namespace DX
} // namespace ChiaSystem
