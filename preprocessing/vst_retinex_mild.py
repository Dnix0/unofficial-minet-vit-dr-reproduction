import numpy as np
from PIL import Image, ImageFilter


def pil_to_float_array(image):
    image = image.convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def float_array_to_pil(arr):
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def make_fundus_mask(arr, threshold=0.04):
    gray = arr.mean(axis=2)
    mask = gray > threshold

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))

    mask_arr = np.asarray(mask_img).astype(np.float32) / 255.0
    mask_arr = np.expand_dims(mask_arr, axis=2)

    return mask_arr


def anscombe_vst(arr):
    x = np.clip(arr, 0.0, 1.0)
    x_counts = x * 255.0

    y = 2.0 * np.sqrt(x_counts + 3.0 / 8.0)

    y_min = y.min()
    y_max = y.max()

    if y_max - y_min < 1e-8:
        return x

    y = (y - y_min) / (y_max - y_min)
    return y.astype(np.float32)


def retinex_mild(arr, mask, radius=35):
    img = float_array_to_pil(arr)
    illum = img.filter(ImageFilter.GaussianBlur(radius=radius))
    illum_arr = pil_to_float_array(illum)

    reflectance = arr / (illum_arr + 1e-6)
    reflectance = np.log1p(reflectance)

    # fundus 영역 기준으로만 normalize
    valid = mask[:, :, 0] > 0.2

    if valid.sum() < 10:
        return arr

    r_min = reflectance[valid].min()
    r_max = reflectance[valid].max()

    if r_max - r_min < 1e-8:
        return arr

    reflectance = (reflectance - r_min) / (r_max - r_min)
    reflectance = np.clip(reflectance, 0.0, 1.0)

    return reflectance.astype(np.float32)


def gamma_correction(arr, gamma=0.97):
    arr = np.clip(arr, 0.0, 1.0)
    return np.power(arr, gamma).astype(np.float32)


def green_channel_guided(arr, strength=0.05):
    green = arr[:, :, 1:2]

    g_min = green.min()
    g_max = green.max()

    if g_max - g_min < 1e-8:
        return arr

    green_norm = (green - g_min) / (g_max - g_min)
    green_rgb = np.repeat(green_norm, 3, axis=2)

    out = (1.0 - strength) * arr + strength * green_rgb
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_vst_retinex_mild(
    image,
    vst_alpha=0.20,
    retinex_alpha=0.25,
    gamma_value=0.97,
    green_strength=0.05,
    retinex_radius=35
):
    """
    Mild VST + Retinex preprocessing.
    Strong preprocessing may wash out lesions, so this function blends enhanced image with original image.
    """
    arr = pil_to_float_array(image)
    original = arr.copy()

    mask = make_fundus_mask(arr, threshold=0.04)

    # mild green-channel guidance
    arr = green_channel_guided(arr, strength=green_strength)

    # mild VST blend
    vst_arr = anscombe_vst(arr)
    arr = (1.0 - vst_alpha) * arr + vst_alpha * vst_arr

    # mild Retinex blend
    ret_arr = retinex_mild(arr, mask=mask, radius=retinex_radius)
    arr = (1.0 - retinex_alpha) * arr + retinex_alpha * ret_arr

    # weak gamma correction
    arr = gamma_correction(arr, gamma=gamma_value)

    # preserve black background
    arr = arr * mask + original * (1.0 - mask)

    return float_array_to_pil(arr)
