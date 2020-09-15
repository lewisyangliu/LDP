function [psnr, ssim]  = psnr_index(imGT, imSR)

% Convert to double (with dynamic range 255)
imSR        = double(imSR); 
imGT        = double(imGT); 

% =========================================================================
% Compute Peak signal-to-noise ratio (PSNR)
% =========================================================================
% mse = mean(mean((imSR - imGT).^2, 1), 2);
imdff = imSR - imGT;
imdff = imdff(:);
rmse = sqrt(mean(imdff.^2));
psnr = 20*log10(255/rmse);
% =========================================================================
% Compute Structural similarity index (SSIM index)
% =========================================================================
% [ssim, ~] = ssim_index(imSR, imGT);

% =========================================================================
% Compute information fidelity criterion (IFC)
% =========================================================================
% ifc = ifcvec(imgt, im);

end