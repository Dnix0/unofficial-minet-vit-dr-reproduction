import numpy as np
from PIL import Image, ImageFilter


def pil_to_float_array(image):
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return arr


def float_array_to_pil(arr):
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def anscombe_vst(arr):
    """
    Variance-Stabilizing Transform approximation for Poisson-like noise.
    Input and output range are normalized to [0, 1].
    """
    x = np.clip(arr, 0.0, 1.0)
    x_counts = x * 255.0

    y = 2.0 * np.sqrt(x_counts + 3.0 / 8.0)

    y_min = y.min()
    y_max = y.max()

    if y_max - y_min < 1e-8:
        return x

    y = (y - y_min) / (y_max - y_min)
    return y.astype(np.float32)


def estimate_illumination(arr, radius=15):
    """
    Simple Retinex-like illumination estimation using Gaussian blur.
    """
    img = float_array_to_pil(arr)
    illum = img.filter(ImageFilter.GaussianBlur(radius=radius))
    illum_arr = pil_to_float_array(illum)

    return np.clip(illum_arr, 1e-6, 1.0)


def retinex_enhancement(arr, radius=15, gain=1.0):
    """
    Simplified Retinex correction:
    reflectance = image / illumination
    then normalize.
    """
    illum = estimate_illumination(arr, radius=radius)

    reflectance = arr / (illum + 1e-6)
    reflectance = np.log1p(gain * reflectance)

    r_min = reflectance.min()
    r_max = reflectance.max()

    if r_max - r_min < 1e-8:
        return arr

    reflectance = (reflectance - r_min) / (r_max - r_min)
    return reflectance.astype(np.float32)


def gamma_correction(arr, gamma=0.9):
    """
    Gamma correction.
    gamma < 1 brightens dark regions.
    """
    arr = np.clip(arr, 0.0, 1.0)
    corrected = np.power(arr, gamma)
    return corrected.astype(np.float32)


def green_channel_guided_enhancement(arr, strength=0.15):
    """
    Green channel guided enhancement.
    Fundus images often show vessels and lesions clearly in green channel.
    """
    green = arr[:, :, 1:2]

    g_min = green.min()
    g_max = green.max()

    if g_max - g_min < 1e-8:
        return arr

    green_norm = (green - g_min) / (g_max - g_min)

    enhanced = (1.0 - strength) * arr + strength * np.repeat(green_norm, 3, axis=2)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def apply_vst_retinex(
    image,
    vst=True,
    retinex=True,
    gamma=True,
    green_guided=True,
    retinex_radius=15,
    gamma_value=0.9
):
    """
    Full preprocessing pipeline:
    RGB image -> green-guided enhancement -> VST -> Retinex -> gamma correction.
    """
    arr = pil_to_float_array(image)

    if green_guided:
        arr = green_channel_guided_enhancement(arr, strength=0.15)

    if vst:
        arr = anscombe_vst(arr)

    if retinex:
        arr = retinex_enhancement(arr, radius=retinex_radius, gain=1.0)

    if gamma:
        arr = gamma_correction(arr, gamma=gamma_value)

    return float_array_to_pil(arr)
