#!/usr/bin/env python3
# Fmriprep post processing script

from nilearn.glm.first_level.design_matrix import make_first_level_design_matrix
import nibabel as nb
import numpy as np
import os, pandas, sys, pdb, argparse, copy, scipy
from os.path import join as pjoin
from nilearn import plotting
from nilearn.input_data import NiftiMasker
import csv
import time
import re
import pylab as plt
import seaborn as sns
from nilearn import image
from nipype.algorithms import confounds as nac
from scipy.signal import savgol_filter

parser = argparse.ArgumentParser(description='Function for performing nuisance regression. Saves resulting output '
                                             'nifti file, information about nuisance regressors and motion (html '
                                             'report), and outputs nibabel object containing clean data')
parser.add_argument('--img_file', help='4d nifti img: file path or nibabel object loaded into memory')
parser.add_argument('--mask_file', help='4d nifti img: file path or nibabel object loaded into memory')
parser.add_argument('--tsv_file', help='tsv file containing nuisance regressors to be removed')
parser.add_argument('--out_path', help='output directory for saving new data file')
#parser.add_argument('--col_names',
#                    help='which columns of TSV file to include as nuisance regressors. defaults to ALL columns.',
#                    nargs="+")
# parser.add_argument('--hp_filter', help='frequency cut-off for high pass filter (removing low frequencies). Recommend '
#                                         '.009 Hz')
# parser.add_argument('--lp_filter', help='frequency cut-off for low pass filter (removing high frequencies). Recommend '
#                                         '.1 Hz for non-task data')
# parser.add_argument('--kernel_size', help='Smoothinging kernel for BOLD image, The functional image will be smoothened before confounds are removed from it')
#parser.add_argument('--out_figure_path',
#                    help='output directory for saving figures. Defaults to location of out_path + _figures')

args = parser.parse_args()

img_file = args.img_file
mask_file = args.mask_file
tsv_file = args.tsv_file
out_path = args.out_path
#col_names = args.col_names
# hp_filter = args.hp_filter
# lp_filter = args.lp_filter
# kernel_size = args.kernel_size
out_figure_path = args.out_path


def fast_regression(X, Y):
    #NEW FUNCTION (MAX SPEED)
    # Compute X^T X
    XtX = np.dot(X.T, X)

    # Compute the inverse of X^T X
    XtX_inv = np.linalg.inv(XtX)

    # Compute X^T Y
    XtY = np.dot(X.T, Y)

    # Compute the coefficients β = (X^T X)^{-1} X^T Y
    beta = np.dot(XtX_inv, XtY)

    # Compute the predictions Ŷ = X β
    Y_hat = np.dot(X, beta)

    # Compute the residuals
    residuals = Y - Y_hat
    return residuals


def denoise(img_file, mask_file, tsv_file, out_path, col_names=False, hp_filter=False, lp_filter=False, kernel_size = False, out_figure_path=False):
    
    print("Yo the script is running")
    subs = '_nosmooth_gsr'
    nii_ext = '.nii.gz'
    FD_thr = [.5]
    sc_range = np.arange(-1, 3)
    constant = 'constant'
    kernel_size = 6
    hp_filter = 0.0078
    lp_filter = 0.1

    # read in files
    # Smoothening the Image using the kernel size
    t1 = time.time()
    raw_img = nb.load(img_file)
    kernel_size = int(kernel_size)
    t2 = time.time()
    print(f"Time to load the image {t2 - t1}")
    print(type(kernel_size), type(raw_img))
    
    # Start Smoothing
    img = raw_img
    data = img.get_fdata()
    print(f"-----------------------------------------------------------------------------------")
    print(img_file)
    print(mask_file)
    print(tsv_file)
    print(out_path)
    df_orig = pandas.read_csv(tsv_file, sep = '\t', na_values='n/a')
    df = copy.deepcopy(df_orig)
    print(f"The shape of the dataframe is {df.shape}")
    print(f"Shape is {data.shape} --------------------------------------------------------------")

    #if int(kernel_size):
    #    img = image.smooth_img(raw_img, kernel_size)
    #else:
    #    print(f"The Kernel size should be an integer")

    ####################
    # UnCOMMENT
    ####################

    # t3 = time.time()
    # print(f"Time to Smoothen the image {t3 - t2}")
    # # Excluding everything other than the brain mask
    # # Loading brain mask
    # brain_mask = nb.load(mask_file)
    # mask_data = brain_mask.get_fdata()
    # # Using the Brain masks and removing eveything except the brain
    # mask_data = mask_data.astype(bool)
    # t4 = time.time()
    # print(f"Time for Brain Masking {t4 - t3}")

    # # get file info
    # img_name = os.path.basename(raw_img.get_filename())
    # file_base = img_name[0:img_name.find('.')]

    # #Finding the subject name
    # # mat = re.search(r'sub-(\d+)', img_name)
    # # print(f"Currently processing subject number {mat.group(1)}-------------------------------------------------------")

    # # You can change the name here to make it easier to recognize your file:
    # save_img_file = pjoin(out_path, file_base + \
    #                       subs + nii_ext)
    # data = img.get_fdata()
    # df_orig = pandas.read_csv(tsv_file, sep = '\t', na_values='n/a')
    # df = copy.deepcopy(df_orig)
    # Ntrs = df.values.shape[0]
    # print('# of TRs: ' + str(Ntrs))
    # assert Ntrs == data.shape[len(data.shape) - 1], "Shape of data does not match TRs"

    # # select columns to use as nuisance regressors
    # conf_names = [
    # "trans_x",
    # "trans_y",
    # "trans_z",
    # "rot_x",
    # "rot_y",
    # "rot_z",
    # "trans_x_derivative1",
    # "trans_y_derivative1",
    # "trans_z_derivative1",
    # "rot_x_derivative1",
    # "rot_y_derivative1",
    # "rot_z_derivative1",
    # "framewise_displacement", #Including FD as a regressor or using it to censor high-motion frames can help mitigate motion-related artifacts
    # "global_signal",
    # "global_signal_derivative1",
    # "csf",
    # "white_matter",
    # "a_comp_cor_00", #Anatomical CompCor (aCompCor) components derived from white matter and CSF are highly effective at removing physiological noise. Including the first 5 components is common practice
    # "a_comp_cor_01",
    # "a_comp_cor_02",
    # "a_comp_cor_03",
    # "a_comp_cor_04",
    # "cosine00", #addressing low-frequency scanner drifts. fMRIPrep includes cosine regressors based on a high-pass filter with a default cutoff of 128 seconds. Incorporating these regressors ensures that low-frequency noise (e.g., scanner drifts) is removed during denoising
    # "cosine01",
    # "cosine02",
    # "cosine03",
    # "cosine04",
    # "cosine05",
    # "cosine06",
    # "cosine07",
    # "cosine08",
    # "cosine09",
    # "cosine10"
    # ]
    # if col_names:
    #     df = df[conf_names] #Changed df.loc[:,col_names]
    #     str_append = '  [SELECTED regressors in CSV]'
    # else:
    #     col_names = df.columns.tolist()
    #     str_append = '  [ALL regressors in CSV]'

    # # fill in missing nuisance values with mean for that variable
    # for col in df.columns:
    #     if sum(df[col].isnull()) > 0:
    #         print('Filling in ' + str(sum(df[col].isnull())) + ' NaN value for ' + col)
    #         df[col] = df[col].fillna(np.mean(df[col]))
    # print('# of Confound Regressors: ' + str(len(df.columns)) + str_append)

    # # implement HP filter in regression
    # TR = img.header.get_zooms()[-1]
    # frame_times = np.arange(Ntrs) * TR
    # if hp_filter:
    #     hp_filter = float(hp_filter)
    #     assert (hp_filter > 0)
    #     df = make_first_level_design_matrix(frame_times, high_pass=hp_filter, add_regs=df.values,
    #                             add_reg_names=df.columns.tolist())
    #     # fn adds intercept into dm

    #     hp_cols = [col for col in df.columns if 'drift' in col]
    #     print('# of High-pass Filter Regressors: ' + str(len(hp_cols)))
    # else:
    #     # add in intercept column into data frame
    #     df[constant] = 1
    #     print('No High-pass Filter Applied')

    # dm = df.values
    # t4 = time.time()
    # print(f"Time for DF preparation {t4 - t3}")
    # # prep data
    # data = np.reshape(data, (-1, Ntrs))
    # data_mean = np.mean(data, axis=1)
    # Nvox = len(data_mean)
    # t5 = time.time()
    # print(f"Time to reshape the image {t5 - t4}")

    # # setup and run regression
    # # model = regression.OLSModel(dm)
    # # results = model.fit(data.T)
    # # if not hp_filter:
    # #     results_orig_resid = copy.deepcopy(results.residuals)  # save for rsquared computation

    # #NEW FUNCTION (MAX SPEED)
    # residuals = fast_regression(dm, data.T)

    # t6 = time.time()
    # print(f"FAST OLS {t6 - t5}")


    # # apply low-pass filter
    # if lp_filter:
    #     # input to savgol_filter is time x voxels
    #     low_pass = float(lp_filter)
    #     Fs = 1. / TR
    #     if low_pass >= Fs / 2:
    #         raise ValueError('Low pass filter cutoff is too close to the Nyquist frequency (%s)' % (Fs / 2))

    #     # Calculate window length for Savitzky-Golay filter
    #     window_length = int(Fs / low_pass)  # Window length in samples
    #     if window_length % 2 == 0:
    #         window_length += 1  # Ensure window length is odd

    #     polyorder = 2  # Polynomial order for the Savitzky-Golay filter

    #     # Apply Savitzky-Golay filter along the time axis (axis=0)
    #     residuals_smoothed = np.apply_along_axis(
    #         savgol_filter, axis=0, arr=residuals, window_length=window_length, polyorder=polyorder
    #     )
        
    #     # Replace original residuals with smoothed residuals
    #     residuals = residuals_smoothed

    #     print('Low-pass Filter Applied: < ' + str(low_pass) + ' Hz')

    # t7 = time.time()
    # print(f"Low pass filter {t7 - t6}")

    # # add mean back into data
    # clean_data = residuals.T + np.reshape(data_mean, (Nvox, 1))  # add mean back into residuals
    # t8 = time.time()
    # print(f"Clean data {t8 - t7}")

    # # save out new data file
    # print('Saving output file...')
    # clean_data = np.reshape(clean_data, img.shape).astype('float32')
    # t9 = time.time()
    # print(f"Reshape clean data {t9 - t8}")

    # # Apply the mask to each 3D volume in the 4D fMRI data
    # masked_clean_data = clean_data * mask_data[:, :, :, np.newaxis]

    # new_img = nb.Nifti1Image(masked_clean_data, img.affine, header=img.header)
    # print(f"the file is getting saved here {save_img_file}")
    # new_img.to_filename(save_img_file)
    # t10 = time.time()
    # print(f"Saving the image {t10 - t9}")


denoise(img_file, mask_file, tsv_file, out_path, out_figure_path)  # No col_names, hp_filter, lp_filter, kernel_size
