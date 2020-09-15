%% example (MATLAB 2017b)
gt = imread('gt.png');
predicted = imread('predicted.png');
PSNR = psnr_index(gt, predicted);
SSIM = ssim_index(gt, predicted);
