import numpy as np
from astropy.io import fits
from cut_resize_tools import *
from tqdm import tqdm
# from Make_Data_Tools import resize, gaussian_filter, remove_nan

# Constants
width = 60
VSmooth = 5
Thresh = 1
Sigma = 2
Sch_RMS = 10
Ech_RMS = 90
Sch_II = 121
Ech_II = 241
Percentile = 99.997
Sigma_Multiply = None

Cut_size_list = [256, 128, 64]
Integrate_layer_num = 30
Obj_size = 100
Maximum_mode = "percentile"
FITS_PATH = "/home/filament/fujimoto/Cygnus-X_CAE/fits/Cygnus_sp16_vs-40_ve040_dv0.25_12CO_Tmb.fits"
OUTPUT_DIR = "/home/filament/fujimoto/Cygnus-X_CAE/data/zroing_resize_data/resize_data/truncation_3size/vflip/"
OUTPUT_FILE_NAME = "CygnusX_v_center"

def process_fits_data(fits_path, output_file_name, cut_size_list, sch_ii, ech_ii, vsmooth, thresh, sigma, sch_rms, ech_rms, integrate_layer_num, percentile, output_dir, obj_size, maximum_mode, sigma_multiply):
    # Load FITS file
    with fits.open(fits_path, memmap=True) as hdul:
        raw_data = hdul[0].data
        # header = hdul[0].header # Not used

        if maximum_mode in ["percentile", "sigma"]:
            max_thresh = maximum_value_determination(
                                                     mode=maximum_mode, 
                                                     data=raw_data, 
                                                     vsmooth=vsmooth, 
                                                     sch_rms=sch_rms, 
                                                     ech_rms=ech_rms, 
                                                     sch_ii=sch_ii, 
                                                     ech_ii=ech_ii, 
                                                     sigma=sigma, 
                                                     thresh=thresh,    
                                                     integrate_layer_num=integrate_layer_num, 
                                                     percentile=percentile
                                                    )
        elif maximum_mode == "normal":
            pass

        else:
            print("maximum_mode must be \"percentile\" or \"sigma\" or \"normal\"")

        for pix in cut_size_list:
            print(f"Processing data clipped to {pix} pixels...")

            # Step1: スライス
            cut_data = slide(raw_data, pix+4)
            print(f"Number of data clipped to {pix} pixels: {len(cut_data)}")

            cut_data = remove_nan(cut_data)
            print(f"Number of data after deletion: {len(cut_data)}")

            # Step2: 並列前処理
            processed_list = parallel_processing(
                                                function=process_data_segment_center_ver, 
                                                target=cut_data,
                                                vsmooth=vsmooth,
                                                sch_rms=sch_rms, 
                                                ech_rms=ech_rms,
                                                sigma=sigma, 
                                                thresh=thresh, 
                                                width=width,
                                                integrate_layer_num=integrate_layer_num
                                                # integrate_layer_num=len(cut_data[0])
                                                )
            del cut_data

            print("Start convolution, resizing and changing max value")
            conv_resize_list = []
            for _data in tqdm(processed_list):
                # _data = select_conv(_data, obj_size, obj_sig)
                _data = resize(_data, (obj_size+4, obj_size+4))
                _data = gaussian_filter(_data)
                if maximum_mode in ["percentile", "sigma"]:
                    _data = np.clip(_data, a_min=None, a_max=max_thresh)
                conv_resize_list.append(_data)
            del processed_list

            if maximum_mode == "normal":
                conv_resize_list = normalization(conv_resize_list)
            elif maximum_mode == "percentile":
                conv_resize_list = normalization_thresh(conv_resize_list, max_thresh)
            elif maximum_mode == "sigma":
                conv_resize_list = normalization_sigma(conv_resize_list, max_thresh, sigma_multiply)
               
            output_file = f"{output_dir}{output_file_name}_mode_{maximum_mode}_{obj_size}x{obj_size}"
            proccess_npyAppend(output_file, conv_resize_list)
            del conv_resize_list
            print(f"Data saved to {output_file}")

def main():
    process_fits_data(
                    fits_path=FITS_PATH,
                    output_file_name=OUTPUT_FILE_NAME,
                    cut_size_list=Cut_size_list,
                    sch_ii=Sch_II,
                    ech_ii=Ech_II,
                    vsmooth=VSmooth,
                    thresh=Thresh,
                    sigma=Sigma,
                    sch_rms=Sch_RMS,
                    ech_rms=Ech_RMS,
                    obj_size=Obj_size,
                    output_dir=OUTPUT_DIR,
                    integrate_layer_num=Integrate_layer_num,
                    percentile=Percentile,
                    maximum_mode=Maximum_mode,
                    sigma_multiply=Sigma_Multiply
    )

if __name__ == "__main__":
    main()