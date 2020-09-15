# LDP
Source code for "Learning Deep Priors for Image Dehazing", ICCV 2019

Dependencies:
Python = 3.6
PyTorch = 0.4.1
torchvision == 0.2.0

Command line for training and test:
Please see ./code/demo.sh

Datasets for training and test:
1). Traning:
|--dataset  
    |--train  
        |--haze
            |--1.png
            |--2.png
                ：  
	|--A
            |--1.png
            |--2.png
                ：  
	|--t
            |--1.png
            |--2.png
                ：  
	|--latent
            |--1.png
            |--2.png
                ：  
    |--test (validation dataset)  
        |--haze
            |--1.png
            |--2.png
                ：  
	|--A
            |--1.png
            |--2.png
                ：  
	|--t
            |--1.png
            |--2.png
                ：  
	|--latent
            |--1.png
            |--2.png
                ：  
2). Test:
|--dataset  
    |--train  
        |--haze
            |--1.png
            |--2.png
                ：  
	|--latent
            |--1.png
            |--2.png
                ： 

Code for computing PSNR and SSIM:
Please see ./cal_psnr_ssim

Citation:
@InProceedings{Liu_2019_ICCV,
          author = {Liu, Yang and Pan, Jinshan and Ren, Jimmy and Su, Zhixun},
          title = {Learning Deep Priors for Image Dehazing},
          booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
          month = {October},
          year = {2019}
          }
