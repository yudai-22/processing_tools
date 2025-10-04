import numpy as np
import matplotlib.pyplot as plt
import copy

import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
import aplpy



def v2ch(v, w): # v(km/s)をchに変える
    x_tempo, y_tempo, v_tempo   = w.wcs_pix2world(0, 0, 0, 0)
    x_ch, y_ch, v_ch   = w.wcs_world2pix(x_tempo, y_tempo, v*1000.0, 0)
    v_ch = int(round(float(v_ch), 0))
    return v_ch


def ch2v(ch, w):  # チャンネル番号を視線速度 (v: km/s) に変換
    x_tempo, y_tempo, v_tempo = w.wcs_pix2world(0, 0, ch, 0)
    v_kms = v_tempo / 1000.0  # m/s を km/s に変換
    return v_kms


def galactic_to_icrs(lon, lat):  # 銀経銀緯を赤経赤緯へ
    coord_galactic = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame='galactic')  # 銀河座標系
    icrs = coord_galactic.icrs  # ICRS座標系(J2000)
    return icrs.ra.deg, icrs.dec.deg


def icrs_to_galactic(ra, dec):#赤経赤緯を銀経銀緯へ
    coord_icrs = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')  # ICRS座標系(J2000)
    galactic = coord_icrs.galactic
    return galactic.l.deg, galactic.b.deg


def del_header_key(header, keys): # headerのkeyを消す
    import copy
    h = copy.deepcopy(header)
    for k in keys:
        try:
            del h[k]
        except:
            pass
    return h


def make_new_hdu_integ(fits_path, v_start_ch, v_end_ch): # 指定速度積分強度のhduを作る
    hdu = fits.open(fits_path)[0]
    data = hdu.data
    header = hdu.header
    
    w = WCS(fits_path)
    
    new_data = np.nansum(data[v_start_ch:v_end_ch+1], axis=0)*np.abs(header["CDELT3"])/1000.0
    header = del_header_key(header, ["CRVAL3", "CRPIX3", "CRVAL3", "CDELT3", "CUNIT3", "CTYPE3", "CROTA3", "NAXIS3", "PC1_3", "PC2_3", "PC3_3", "PC3_1", "PC3_2"])
    header["NAXIS"] = 2
    new_hdu = fits.PrimaryHDU(new_data, header)
    return new_hdu


def plot_selected_channel(data, start_ch=None, end_ch=None, tittle=None, grid=50, savefig = False):#data_shape=(depth, width, height)
    plt.figure(figsize=(6, 6))
    mean_data = np.nanmean(data, axis=(1, 2))
    
    plt.plot(np.arange(len(mean_data)), mean_data, "k")
    plt.xlabel("Channel")
    plt.xticks(np.arange(0, len(mean_data), grid), fontsize=8)
    plt.ylabel("Mean Intensity [K]")

    if start_ch is not None:
        plt.axvline(x=start_ch, color='red', linestyle='--', label=f'channel = {start_ch}')
    if end_ch is not None:
        plt.axvline(x=end_ch, color='b', linestyle='--', label=f'channel = {end_ch}')

    plt.grid()
    plt.legend()
    plt.title(str(tittle), fontsize=14)
    
    if savefig == True:
        plt.savefig(f"{str(tittle)}.png")

    plt.show()


def astro_image(hdu):#図の描画(aplpy)
    gc_distance = 1400
    scale_per_pc = np.rad2deg(np.arcsin(1/gc_distance))
    print(scale_per_pc)
    
    fig=aplpy.FITSFigure(hdu)
    
    fig.show_colorscale(cmap="nipy_spectral", stretch='linear')
    fig.set_nan_color("black")
    fig.add_colorbar()
    fig.colorbar.set_location('right')
    fig.colorbar.set_axis_label_text("[K km s⁻¹]")
    fig.colorbar.set_axis_label_font(size=15)
    
    # fig.tick_labels.set_xformat("dd.d")
    fig.tick_labels.set_font(size=15) 
    fig.ticks.set_color("black")
    fig.axis_labels.set_font(size=15)
    
    fig.add_scalebar(scale_per_pc*10) # スケールバーの長さ（単位は座標系に依存）
    fig.scalebar.set_label('10pc')
    fig.scalebar.set_color('w')
    fig.scalebar.set_linewidth(2)
    fig.scalebar.set_font(size=15) 
    
    # plt.savefig("Cygnus-X_NRO45m.png")


def calculate_resolution(wave_lambda, telescope_diameter):
    telescope_resolution = np.rad2deg(wave_lambda / telescope_diameter) * 60
    return telescope_resolution


def create_smoothing_kernel(original_resolution, pixel_grid, target_resolution):
    """
    画像を目標の分解能に平滑化するためのガウシアンカーネルを作成する。

    Parameters
    ----------
    original_resolution : float
        元の画像の分解能（FWHM）、単位は秒角。
    pixel_grid : float
        1ピクセルの大きさ、単位は秒角/ピクセル。
    target_resolution : float
        目標とする最終的な分解能（FWHM）、単位は秒角。

    Returns
    -------
    astropy.convolution.Kernel2D
        畳み込みに使用する2Dガウシアンカーネル。
    """
    if target_resolution < original_resolution:
        raise ValueError("目標の分解能は元の分解能より大きい必要があります。")

    # 必要なカーネルの分解能を計算（二乗和の平方根）
    # R_final^2 = R_orig^2 + R_kernel^2
    kernel_fwhm_arcsec = np.sqrt(target_resolution**2 - original_resolution**2)

    # カーネルの分解能をピクセル単位に変換
    kernel_fwhm_pixels = kernel_fwhm_arcsec / pixel_grid

    # FWHMからガウス関数の標準偏差（sigma）に変換
    # FWHM = 2.355 * sigma
    sigma_pixels = kernel_fwhm_pixels / (2 * np.sqrt(2 * np.log(2)))

    # 計算したsigma値を持つガウシアンカーネルを生成
    kernel = Gaussian2DKernel(x_stddev=sigma_pixels, y_stddev=sigma_pixels)
    
    # カーネルを正規化（合計が1になるように）
    kernel.normalize('integral')
    
    return kernel
