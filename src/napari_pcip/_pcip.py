# -*- coding: utf-8 -*-

"""
pcip.py:
Polarisation camera image processing
The script provides utility functions that can be imported to process images
from a polarisation camera with four pol channels.
GPU acceleration using CuPy is used where possible.
"""

import os
import sys
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb as mpl_hsv_to_rgb

use_cupy = True

if use_cupy:
    try:
        import cupy as xp
        import cupy.fft as fft
        from cupyx.scipy.ndimage import gaussian_filter
        from cupyx.scipy.ndimage import shift as spshift
        print('CuPy detected')
    except ModuleNotFoundError:
        import numpy as xp
        import numpy.fft as fft
        from scipy.ndimage import gaussian_filter
        from scipy.ndimage import shift as spshift
        print('CuPy not detected')
        use_cupy = False
else:
    import numpy as xp
    import numpy.fft as fft
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as spshift
    print('CuPy not used')


__author__ = "Zhonghe Feng"
__version__ = "1.7.1"


# disable prints, ref:
# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def gpu_io(func):
    """
    Decorator which allows an array to be passed to and from GPU where
    appropriate.
    """
    def inner(img, *args, use_cupy=use_cupy, opt_fmt='cp', **kwargs):
        assert opt_fmt in ('cp', 'np'), 'opt_fmt must be "cp" or "np"!'
        # convert numpy to cupy arr when necessary
        if use_cupy and not isinstance(img, xp._core.ndarray):
            img = xp.asarray(img)
        # execute the original func
        result = func(img, *args, **kwargs)
        # convert cupy to numpy arr when necessary
        if use_cupy and opt_fmt != 'cp':
            result = result.get()
        return result
    return inner


@gpu_io
def pol_split(img):
    """
    Split a raw image from a polarisation camera into 4 images corresponding to
    each of the pol channels, in order of 0, 45, -45, 90 degrees.
    """
    print('Start splitting polarisation channels...')
    y, x = img.shape
    assert y % 2 == x % 2 == 0, 'Please pass in a correct image!'
    y, x = img.shape
    y_half = y // 2
    x_half = x // 2
    results = xp.zeros((4, y_half, x_half), dtype='int32')
    results[3] = img[::2, ::2]
    results[1] = img[::2, 1::2]
    results[2] = img[1::2, ::2]
    results[0] = img[1::2, 1::2]
    return results


@gpu_io
def ftshift(img, shift, do_filter=False, r0=270):
    """
    Implementation of linear translation of a single image using Fourier
    shift theorem. Usage similar to scipy.ndimage.shift with mode='wrap'.
    It however assumes the input is a real-valued array.
    do_filter: do a Fourier filtering to enhance spatial frequencies
        characterised by r0.
    """
    # rfft
    img = fft.fftshift(img)
    FT_img = fft.rfft2(img)
    FT_img = fft.fftshift(FT_img, axes=0)
    # shift
    FT_x_len = FT_img.shape[0]  # extract array shape
    FT_y_len = FT_img.shape[1]
    FT_x = xp.arange(FT_x_len) - FT_x_len // 2  # linear coordinates in Fourier
    FT_y = xp.arange(FT_y_len)
    # construct complex exponentials
    exp_x = xp.exp(-1j * 2 * xp.pi * FT_x * shift[0] / FT_x_len)
    exp_y = xp.exp(-1j * 2 * xp.pi * FT_y * shift[1] / FT_y_len / 2)
    exp_arr = xp.outer(exp_x, exp_y)
    FT_shifted_img = FT_img * exp_arr  # multiply by complex exponential
    # do filtering
    if do_filter:
        FT_X, FT_Y = xp.meshgrid(FT_x, FT_y, indexing='ij')
        r2 = (FT_X / r0) ** 2 + (FT_Y * FT_x_len / FT_y_len / 2 / r0) ** 2
        mask = 1 / ((1 + r2) ** 2 - 3 * r2) ** 3
        FT_shifted_img *= mask[:,:]
    # irfft
    FT_shifted_img = fft.ifftshift(FT_shifted_img, axes=0)
    shifted_img = fft.irfft2(FT_shifted_img)
    shifted_img = fft.ifftshift(shifted_img)
    return shifted_img


@gpu_io
def pshift(img, method='ft', do_filter=False, r0=270):
    """
    Sub-pixel shifting of the polarisation channels into their correct physical
    positions.
    When the method is 'ft', it uses Fourier shift theorem for shifting;
    when the method is 'sp', it uses scipy.ndimage.shift or
    cupyx.scipy.ndimage.shift to shift the pixels.
    Return a stack of arrays corresponding to each of the polarisation
    channels, but a ring whose width is 1 pixel is removed at the boundary of
    each image.
    do_filter: do a Fourier filtering to enhance spatial frequencies
        characterised by r0. If set to True, it would override the method
        option and always use 'ft' as the method
    """
    assert method in ('ft', 'sp'), 'Method not recognised!'
    if method == 'ft' or do_filter:
        if do_filter:
            def shift(*args):
                return ftshift(*args, do_filter=True, r0=r0)
        else:
            shift = ftshift
    elif method == 'sp':
        shift = spshift
    img[3] = shift(img[3], (.25, .25))
    img[1] = shift(img[1], (.25, -.25))
    img[2] = shift(img[2], (-.25, .25))
    img[0] = shift(img[0], (-.25, -.25))
    # set the outer ring to zeros
    img[:, -1:1, :] = 0
    img[:, :, -1:1] = 0
    return img
    return img[:, 1:-1, 1:-1]


@gpu_io
def sub_samp_single(img, mag=2):
    """
    Sub-sample an image by zero-padding at the Fourier space.
    img: must be a real-valued array
    mag: magnification
    """
    # rfft
    img = fft.fftshift(img)
    FT_img = fft.rfft2(img)
    FT_img = fft.fftshift(FT_img, axes=0)
    # zero padding
    width_x = img.shape[0] * (mag - 1) // 2  # along the first axis
    width_y = img.shape[1] * (mag - 1) // 2
    pad_width = ((width_x, width_x), (0, width_y))
    FT_padded_img = xp.pad(FT_img, pad_width)
    # irfft
    FT_padded_img = fft.ifftshift(FT_padded_img, axes=0)
    sub_img = fft.irfft2(FT_padded_img)
    sub_img = fft.ifftshift(sub_img)
    return sub_img * mag * mag


@gpu_io
def sub_samp_stack(img, mag=2, do_filter=False, r0=270):
    """
    Sub-sample a stack of iamges by zero-padding at the Fourier space.
    img: must be a stack of images which are real
    mag: magnification
    do_filter: do a Fourier filtering to enhance spatial frequencies
        characterised by r0.
    """
    # rfft
    img = fft.fftshift(img, axes=(1, 2))
    FT_img = fft.rfft2(img)
    FT_img = fft.fftshift(FT_img, axes=1)
    # construct FT coordinates
    Nx = FT_img.shape[1]
    Ny = FT_img.shape[2]
    FT_x = xp.arange(Nx) - Nx // 2  # linear coordinates in Fourier
    FT_y = xp.arange(Ny)
    # do filtering
    if do_filter:
        FT_X, FT_Y = xp.meshgrid(FT_x, FT_y, indexing='ij')
        r2 = (FT_X / r0) ** 2 + (FT_Y * Nx / Ny / 2 / r0) ** 2
        mask = 1 / ((1 + r2) ** 2 - 3 * r2) ** 3
        FT_img *= mask[xp.newaxis, :, :]
    # zero padding
    width_x = img.shape[1] * (mag - 1) // 2  # along the first axis
    width_y = img.shape[2] * (mag - 1) // 2
    pad_width = ((0, 0), (width_x, width_x), (0, width_y))
    FT_padded_img = xp.pad(FT_img, pad_width)
    # irfft
    FT_padded_img = fft.ifftshift(FT_padded_img, axes=1)
    sub_img = fft.irfft2(FT_padded_img)
    sub_img = fft.ifftshift(sub_img, axes=(1, 2))
    return sub_img * mag * mag


@gpu_io
def pshift_and_subsamp(img, mag=2, do_filter=False, r0=270):
    """
    Combine pshift and sub-sampling on a stack of images for 4
    polarisation channels.
    do_filter: do a Fourier filtering to enhance spatial frequencies
        characterised by r0.
    """
    # rfft
    img = fft.fftshift(img, axes=(1, 2))
    FT_img = fft.rfft2(img)
    FT_img = fft.fftshift(FT_img, axes=1)
    # construct FT coordinates
    Nx = FT_img.shape[1]
    Ny = FT_img.shape[2]
    FT_x = xp.arange(Nx) - Nx // 2  # linear coordinates in Fourier
    FT_y = xp.arange(Ny)
    # define shift amounts
    # shiftx = (-.25, .25, -.25, .25)
    # shifty = (-.25, -.25, .25, .25)
    # construct 1d exponential array
    exp_x_p = xp.exp(-1j * 2 * xp.pi * FT_x / Nx * .25)
    exp_y_p = xp.exp(-1j * xp.pi * FT_y / Ny * .25)
    exp_x_n = xp.conjugate(exp_x_p)
    exp_y_n = xp.conjugate(exp_y_p)
    # extend to 2d exponential for x and y respectively
    exp_x_2d = xp.stack((exp_x_n, exp_x_p, exp_x_n, exp_x_p),)
    exp_y_2d = xp.stack((exp_y_n, exp_y_n, exp_y_p, exp_y_p),)
    # extend to 3d exponential array using vectorised outer product
    exp_3d = exp_x_2d[:, :, xp.newaxis] * exp_y_2d[:, xp.newaxis, :]
    # apply the exponential array
    FT_img *= exp_3d
    # do filtering
    if do_filter:
        FT_X, FT_Y = xp.meshgrid(FT_x, FT_y, indexing='ij')
        r2 = (FT_X / r0) ** 2 + (FT_Y * Nx / Ny / 2 / r0) ** 2
        mask = 1 / ((1 + r2) ** 2 - 3 * r2) ** 3
        FT_img *= mask[xp.newaxis, :, :]
    # zero padding
    width_x = Nx * (mag - 1) // 2  # along the first axis
    width_y = int(Ny * (mag - 1))
    pad_width = ((0, 0), (width_x, width_x), (0, width_y))
    FT_padded_img = xp.pad(FT_img, pad_width)
    # irfft
    FT_padded_img = fft.ifftshift(FT_padded_img, axes=1)
    sub_img = fft.irfft2(FT_padded_img)
    sub_img = fft.ifftshift(sub_img, axes=(1, 2))
    sub_img *= mag * mag
    return sub_img


@gpu_io
def pol_to_stoke(img):
    """
    Calculate the stokes parameters S0, S1, S2 of a given frame.
    img: a stack of 4 images, corresponding to each of the polarisation
    channels, in the order of 0, 45, -45 and 90 degrees.
    """
    assert len(img) == 4, "Input format incorrect: need a stack of 4."
    # initialise output array: int32 would overflow
    stokes = xp.zeros((3, img.shape[1], img.shape[2]), dtype='int64')
    stokes[0] = xp.sum(img, axis=0) / 2
    stokes[1] = img[0] - img[3]
    stokes[2] = img[1] - img[2]
    return stokes


@gpu_io
def stoke_to_hsv(stokes, typ='normal', c4=False):
    """
    Generate the HSV arrays of false colour images from the stokes parameters
    S0, S1 and S2.
    c4: if True, c4 colour wheel (defined from 0 to pi/2) is used;
    if False, c2 colour wheel (defined from 0 to pi) is used.
    output: H, S, V images as an array in the shape of (Ny, Nx, 3).
    """
    assert len(stokes) == 3, "Input format incorrect: need a stack of 3."
    assert typ in ('normal', 'data'), "False colour scheme not recognised"
    h = xp.arctan2(stokes[2], stokes[1]) / (2 * xp.pi)  # [-0.5, 0.5)
    h = (h + 1) % 1  # [0, 1)
    if c4:
        h = (2 * h) % 1
    if typ == 'normal':
        s = xp.sqrt(stokes[2] ** 2 + stokes[1] ** 2) / stokes[0]
        Imax = xp.max(stokes[0])
        v = stokes[0] / Imax
    elif typ == 'data':
        s = xp.full_like(stokes[0], 1)
        v = xp.sqrt(stokes[2] ** 2 + stokes[1] ** 2) / stokes[0]
        v /= xp.max(v)
    hsv = xp.dstack((h, s, v),)
    return hsv


def hsv_to_rgb(hsv):
    """
    Convert hsv array of shape (Ny, Nx, 3) to an rgb array of the same shape
    Input can be np/cp array, but output will always be np array.
    """
    if hasattr(hsv, 'get'):  # if hsv is a cupy array
        hsv = hsv.get()
    rgb = mpl_hsv_to_rgb(hsv)
    return rgb


# cannot use @gpu_io because return value is a tuple
def hsv_hist(img, H_binN=500, S_binN=500):
    """
    Compute a histogram array from an HSV image.
    Can accept img as a numpy or cupy array.
    Histogram in Hue-Saturation space, not S1-S2 Stokes space.
    """
    if use_cupy:
        img = xp.asarray(img)
    # produce bins
    H_bin = xp.linspace(0, 1, H_binN + 1)
    S_bin = xp.linspace(0, 1, S_binN + 1)
    # unpack HSV image
    H = img[:, :, 0].flatten()
    S = img[:, :, 1].flatten()
    V = img[:, :, 2].flatten()
    # produce histogram array
    hist = xp.histogram2d(H, S, bins=(H_bin, S_bin), weights=V)
    return hist


def plot_polar_hist(hist, log1p=True):
    """
    Accept the hist result in hsv_hist and plot a polar histogram in matplotlib
    """
    # unpack hist result
    values = hist[0]
    th = hist[1] * xp.pi * 2
    r = hist[2]
    # th = (hist[1][:-1] + hist[1][1:]) * np.pi  # theta for hue [0, 2pi)
    # r = (hist[2][:-1] + hist[2][1:]) / 2
    R, Th = xp.meshgrid(r, th)
    # normalise values according to the unit area they extend
    values /= R[:-1, :-1] + r[1]
    # adjust value sensitivity
    if log1p:
        values = xp.log1p(values)
    # plot polar pcolourmesh
    plt.figure()
    plt.subplot(projection='polar')
    if hasattr(Th, 'get'):  # cannot check hist because hist is tuple
        Th = Th.get()
        R = R.get()
        values = values.get()
    plt.pcolormesh(Th, R, values)
    plt.grid(lw=.3)  # linewidth 3


# cannot use @gpu_io because return value is a tuple
def stoke_hist(stoke, S1_binN=500, S2_binN=500):
    """
    Compute a histogram array from arrays of stokes parameters.
    Can accept img as a numpy or cupy array.
    Histogram in normalised S1-S2 space, not in Hue-Sat space.
    """
    # convert to cp if necessary
    if use_cupy:
        stoke = xp.asarray(stoke)
    # produce bins
    S1_bin = xp.linspace(-1, 1, S1_binN + 1)
    S2_bin = xp.linspace(-1, 1, S2_binN + 1)
    # normalise and flatten stokes parameters
    S1 = (stoke[1] / stoke[0]).flatten()
    S2 = (stoke[2] / stoke[0]).flatten()  # .flat .ravel
    # S0 = stoke[0].flatten()
    # produce histogram array
    # hist = xp.histogram2d(S1, S2, bins=(S1_bin, S2_bin), weights=S0)
    hist = xp.histogram2d(S1, S2, bins=(S1_bin, S2_bin))
    return hist


def plot_stoke_hist(hist, log1p=True):
    """
    Accept the hist result in stoke_hist and plot a histogram of S1 and S2.
    """
    # unpack hist result
    heights = hist[0]
    s1 = hist[1]
    s2 = hist[2]
    s1, s2 = xp.meshgrid(s1, s2, indexing='ij')
    if log1p:
        heights = xp.log1p(heights)
    if hasattr(s1, 'get'):  # if it is cupy array
        s1 = s1.get()
        s2 = s2.get()
        heights = heights.get()
    # print(s1.shape, s2.shape, heights.shape):(501, 501) (501, 501) (500, 500)
    # masking
    mask = s1 ** 2 + s2 ** 2 > 1  # this is (N+1)*(N+1)
    # do a logical or of all corners of a 2D bin
    mask = mask[1:, 1:] * mask[1:, :1] * mask[:1, 1:] * mask[:1, :1]  # 178 us
    heights = ma.MaskedArray(heights, mask=mask)
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    plt.pcolormesh(s1, s2, heights)
    plt.xlabel('S1 / S0')
    plt.ylabel('S2 / S0')
    ax.axhline(0, c='w', lw=.3, alpha=.5)
    ax.axvline(0, c='w', lw=.3, alpha=.5)
    # add circular grids
    for r in (.2, .4, .6, .8):
        circ = plt.Circle((0, 0), r, alpha=.5, ec='w', fill=False, lw=.3)
        ax.add_patch(circ)
    # add diagonal grids
    # ax.plot((0, 1), (0, 1), transform=ax.transAxes)
    x = np.linspace(-1, 1, len(mask))
    lin_mask = x * x > 1 / 2
    y1 = ma.MaskedArray(x, mask=lin_mask)
    y2 = ma.MaskedArray(-x, mask=lin_mask)
    plt.plot(x, y1, alpha=.5, c='w', lw=.3)
    plt.plot(x, y2, alpha=.5, c='w', lw=.3)


def raw_to_rgb(img, do_pshift=False, pmethod='ft', subsamp=False, mag=4,
               do_filter=False, r_filter=270, bg_rmv=False, scheme='normal',
               c4=False, transposed=False, log1p=False):
    """
    A function to convert a raw polarisation camera image to a false image in
    rgb format.
    img: raw image
    pshift: pixel shifting
    pmethod: 'ft' or 'sp' for Fourier shift or scipy shift
    subsamp: subsampling
    mag: magnification for subsampling
    do_filter: do a Fourier filtering to enhance spatial frequencies
        characterised by r_filter.
    bg_rmv: background removal
    scheme: 'normal', 'data', 'avg' or 'fourier'
    c4: c4 colour wheel. If False, use c2 colour wheel
    log1p: A log1p enhancement for Fourier scheme
    Primarily implemented for real-time false colour image display for
    scopefoundry to replace raw_to_false in false_colour.py.
    """
    with HiddenPrints():  # disable prints
        img = xp.array(img)
        pols = pol_split(img)
        if subsamp and do_pshift:
            pols = pshift_and_subsamp(pols, mag=mag, do_filter=do_filter,
                                      r0=r_filter)
        elif subsamp and not do_pshift:
            pols = sub_samp_stack(pols, mag=mag, do_filter=do_filter,
                                  r0=r_filter)
        elif not subsamp and do_pshift:
            pols = pshift(pols, pmethod, do_filter=do_filter, r0=r_filter)
        stokes = pol_to_stoke(pols)
        if bg_rmv:
            sigma = 400 + 400 * (mag - 1) * subsamp
            for i in (1, 2):  # removal background on S1 and S2 only
                # stokes[i] -= filters.gaussian(stokes[i], sigma)
                stokes[i] -= gaussian_filter(stokes[i], sigma, mode='nearest')
        if scheme in ('avg', 'fourier'):
            if scheme == 'avg':
                opt = stokes[0]
            else:
                assert scheme == 'fourier'
                opt = xp.sqrt(stokes[1] ** 2 + stokes[2] ** 2) / stokes[0]
            if transposed:
                opt = opt.T
            if scheme == 'fourier':
                opt = xp.absolute(fft.fftshift(fft.fft2(fft.fftshift(opt))))
                if log1p:
                    opt = xp.log1p(opt)
            if use_cupy:
                return opt.get()
            else:
                return opt
        hsv = stoke_to_hsv(stokes, c4=c4, typ=scheme)
        rgb = hsv_to_rgb(hsv)  # force to numpy here
        if transposed:
            rgb = np.transpose(rgb, (1, 0, 2))
    return rgb


def raw_to_hist(img, do_pshift=False, pmethod='ft', subsamp=False, mag=4,
                do_filter=False, r_filter=270, bg_rmv=False, log1p=True,
                plot=True):
    """
    A function to compute a raw polarisation camera image into a histogram
    in the S1-S2 space.
    img: raw image
    pshift: pixel shifting
    pmethod: 'ft' or 'sp' for Fourier shift or scipy shift
    subsamp: subsampling
    mag: magnification for subsampling
    bg_rmv: polarisation background removal
    do_filter: do a Fourier filtering to enhance spatial frequencies
        characterised by r_filter.
    """
    with HiddenPrints():  # disable prints
        img = xp.array(img)
        pols = pol_split(img)
        if subsamp and do_pshift:
            pols = pshift_and_subsamp(pols, mag=mag, do_filter=do_filter,
                                      r0=r_filter)
        elif subsamp and not do_pshift:
            pols = sub_samp_stack(pols, mag=mag, do_filter=do_filter,
                                  r0=r_filter)
        elif not subsamp and do_pshift:
            pols = pshift(pols, pmethod, do_filter=do_filter, r0=r_filter)
        stokes = pol_to_stoke(pols)
        if bg_rmv:
            sigma = 400 + 400 * (mag - 1) * subsamp
            for i in (1, 2):  # removal background on S1 and S2 only
                # stokes[i] -= filters.gaussian(stokes[i], sigma)
                stokes[i] -= gaussian_filter(stokes[i], sigma, mode='nearest')
        hist = stoke_hist(stokes)
        if plot:
            plot_stoke_hist(hist, log1p=log1p)
        else:
            return tuple([i.get() for i in hist])

# EOF
