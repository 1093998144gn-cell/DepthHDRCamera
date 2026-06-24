import torch

def conv_fft(img_c, kernel_c, pad_width=None):
    """
    针对物理模拟优化的FFT卷积
    """
    # 确保输入是 float32 (有些 mat 加载出来可能是 float64)
    img_c = img_c.float()
    kernel_c = kernel_c.float()

    if pad_width is not None:
        img_c_pad = torch.nn.functional.pad(img_c, pad_width)
        kernel_c_pad = torch.nn.functional.pad(kernel_c, pad_width)
    else:
        img_c_pad = img_c
        kernel_c_pad = kernel_c

    # 1. 执行FFT
    # 显式使用 rfft2 (针对实数输入的优化) 或者确保 fft2 后取实部
    img_fft = torch.fft.fft2(img_c_pad, dim=(-2, -1))
    kernel_fft = torch.fft.fft2(kernel_c_pad, dim=(-2, -1))

    # 2. 频域相乘
    # 检查 kernel 是否包含虚部，物理上 PSF 是强度(实数)，但传播过程是复数振幅
    # 如果 kernel 是强度，这里直接乘；如果是复振幅，需要取模
    conv_result_fft = img_fft * kernel_fft

    # 3. 逆FFT并取实部
    # 这是最关键的一步，必须使用 .real 并且确保不是 ifft (旧版)
    # .abs() 有时比 .real 更稳健，因为物理能量总是正的
    im_conv = torch.fft.ifft2(conv_result_fft, dim=(-2, -1)).real

    # 4. 裁剪
    if pad_width is not None:
        # 注意这里的索引顺序 (left, right, top, bottom) -> [..., top:-bottom, left:-right]
        im_conv = im_conv[..., pad_width[2]:-pad_width[3], pad_width[0]:-pad_width[1]]

    # 5. 最后的数值安全检查：防止产生微小的负数导致绘图全黑
    im_conv = torch.clamp(im_conv, min=0.0)
    
    return im_conv