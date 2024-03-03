import numpy as np
import cv2
import random
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

import time


def crop_images_to_common_min_size(images):

    min_width = float('inf')
    min_height = float('inf')
    if len(images) == 0:
        min_width = 0
        min_height = 0
    else:
        for image in images:
            # print(image)
            height, width = image.shape[0:2]
            min_width = min(min_width, width)
            min_height = min(min_height, height)

    cropped_images = []

    for image in images:
        height, width = image.shape[0:2]

        left = (width - min_width) // 2
        top = (height - min_height) // 2

        right = left + min_width
        bottom = top + min_height

        cropped_image = image[top:bottom, left:right]
        cropped_images.append(cropped_image)

    return cropped_images


def ReadImage(file_name, method='pixel'):
    if method == 'evenly':
        # using crop by evenly dividing by 3
        original_image = cv2.imread('data/'+file_name+'.jpg')
        original_image = original_image[:,:,0]
        height, width = original_image.shape
        channel_height = height // 3
        image_channel1 = original_image[0:channel_height,:]
        image_channel2 = original_image[channel_height:2*channel_height,:]
        image_channel3 = original_image[2*channel_height:3*channel_height,:]

    # #manually crop for original image
    # image_channel1 = cv2.imread('data/'+file_name+'1.jpg')
    # image_channel2 = cv2.imread('data/'+file_name+'2.jpg')
    # image_channel3 = cv2.imread('data/'+file_name+'3.jpg')
    # original_image = cv2.imread('data/'+file_name+'.jpg')

    # image_channel1 = image_channel1[:,:,1]
    # image_channel2 = image_channel2[:,:,1]
    # image_channel3 = image_channel3[:,:,1]

    # manually crop in pixel
    elif method == 'pixel':
        original_image = cv2.imread('data/'+file_name+'.jpg')
        original_image = original_image[:,:,0]
        left = 27
        right = 377
        image_channel1 = original_image[27:342,left:right]
        image_channel2 = original_image[355:680,left:right]
        image_channel3 = original_image[695:1005,left:right]
    
    elif method == 'ps':
        image_channel1 = cv2.imread('data_hires/'+file_name+'1.tif')
        image_channel2 = cv2.imread('data_hires/'+file_name+'2.tif')
        image_channel3 = cv2.imread('data_hires/'+file_name+'3.tif')
        # original_image = cv2.imread('data_hires/'+file_name+'.tif')

        image_channel1 = image_channel1[:,:,1]
        image_channel2 = image_channel2[:,:,1]
        image_channel3 = image_channel3[:,:,1]


    
    image_channel1, image_channel2, image_channel3 = crop_images_to_common_min_size([image_channel1, image_channel2, image_channel3])

    # # show the image of each channel
    # imshowImage(image_channel1)
    # imshowImage(image_channel2)
    # imshowImage(image_channel3)

    return image_channel1, image_channel2, image_channel3



def compute_fourier_alignment_offset(image1, image2, filename, preprocessing=True, firstpair=True):
    # gaussian parameter
    sigma = 1

    width = image1.shape[1]
    height = image1.shape[0]

    if preprocessing:
        # gaussian filter
        image1_filter = gaussian_filter(image1, sigma)
        image2_filter = gaussian_filter(image2, sigma)
        # sharpening the image
        image1 = image1 - image1_filter
        image2 = image2 - image2_filter
    
    # Calculate the cross-correlation manually
    fft_image1 = np.fft.fft2(image1)
    fft_image2 = np.fft.fft2(image2)
    cross_corr = np.fft.ifftshift(np.fft.ifft2(fft_image1 * np.conjugate(fft_image2)))

    # peak
    y_peak, x_peak = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)
    print(y_peak,x_peak)
    
    # displacement of the peak to the centre of the picture
    y_peak -= height // 2
    x_peak -= width // 2
    offset_found =  [x_peak, y_peak]
    print("offset: ", offset_found)


    # aligned_image2 = np.roll(image2, offset_found, axis=(0, 1))

    # # Plot the original and aligned images
    # plt.figure(figsize=(12, 6))

    # plt.subplot(2, 3, 1)
    # plt.imshow(image1, cmap='gray')
    # plt.title('Image 1')

    # plt.subplot(2, 3, 2)
    # plt.imshow(image2, cmap='gray')
    # plt.title('Image 2 (Offset)')

    # plt.subplot(2, 3, 3)
    # plt.imshow(np.abs(cross_corr), cmap='gray')
    # plt.title('Cross-correlation')

    # plt.subplot(2, 3, 4)
    # plt.imshow(aligned_image2, cmap='gray')
    # plt.title('Aligned Image 2')

    # plt.tight_layout()
    # plt.show()

    # Convert cross_corr to a suitable data type (float32 or float64)
    cross_corr = cross_corr.astype(np.float32)
    # Normalize cross_corr for display (optional)
    normalized_corr = cv2.normalize(cross_corr, None, 0, 255, cv2.NORM_MINMAX)
    # Convert cross_corr to 8-bit unsigned integer
    cross_corr_gray = np.uint8(normalized_corr)
    if firstpair:
        if preprocessing:
            cv2.imwrite('inverseFFT_'+filename+'_first_pre.jpg', cross_corr_gray)
        else:
            cv2.imwrite('inverseFFT_'+filename+'_first.jpg', cross_corr_gray)
    else:
        if preprocessing:
            cv2.imwrite('inverseFFT_'+filename+'_second_pre.jpg', cross_corr_gray)
        else:
            cv2.imwrite('inverseFFT_'+filename+'_second.jpg', cross_corr_gray)

    return offset_found


def imshowImage(image, scale_percent=300):
    # 计算新的宽度和高度
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    # 使用cv2.resize来缩放图像
    resized_image = cv2.resize(image, (width, height))
    
    # 将NumPy数组转换为8位无符号整数数据类型（必须是uint8类型）
    cv2.imshow('image',resized_image)

    # 等待用户按下任意键，然后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def ShowOverlapRegion_3channels(image1, image2, image3, offset2, offset3,file_name,scale_percent,output=True):
    dx2 = offset2[0]
    dy2 = offset2[1]
    dx3 = offset3[0]
    dy3 = offset3[1]

    h1, w1 = image1.shape[0:2]
    h2, w2 = image2.shape[0:2]
    h3, w3 = image3.shape[0:2]

    # Calculate the overlap region
    x1 = max(0, dx2, dx3)
    x2 = min(w1, w2 + dx2, w3 + dx3)
    y1 = max(0, dy2, dy3)
    y2 = min(h1, h2 + dy2, h3 + dy3)
    if x1 < x2  and y1 < y2:
        overlap_image1 = image1[y1:y2, x1:x2]
        overlap_image2 = image2[y1 - dy2:y2 - dy2, x1 - dx2:x2 - dx2]
        overlap_image3 = image3[y1 - dy3:y2 - dy3, x1 - dx3:x2 - dx3]
    else:
        print('[Error]no region of overlap!')
        return
    
    # rgb_image = np.zeros((y2-y1,x2-x1,3))
    # rgb_image[:, :, 0] = overlap_image1
    # rgb_image[:, :, 1] = overlap_image2
    # rgb_image[:, :, 2] = overlap_image3
    
    # 合成彩色图像
    merged_image = cv2.merge((overlap_image1, overlap_image2, overlap_image3))    

    # 保存图像
    cv2.imwrite('merged_'+file_name+'.jpg', merged_image)
    print("Image saved successfully.")

    if output:
        imshowImage(merged_image, scale_percent)

    return

def main():
    file_name_list = ['00125v', '00149v', '00153v', '00351v', '00398v', '01112v']
    high_resolution_name_list = ['01047u', '01657u', '01861a']
    file_type = 1 #set to 0 to process normal images, set to 1 for high resolution images
    file_index = 2 #set to -1 to let program to randomly choose a image
    preprocessing = True
    show = False
    print("*********************file_type = ", file_type, "*********************")
    start_time = time.time()

    if not file_type:
        if file_index == -1:
            file_name = random.choice(file_name_list)
        else:
            file_name = file_name_list[file_index]
        image_channel1, image_channel2, image_channel3 = ReadImage(file_name,'pixel')
        print('[Info]Choose file: ', file_name)

        best_offset2 = compute_fourier_alignment_offset(image_channel1, image_channel2, file_name, preprocessing, firstpair=True)
        best_offset3 = compute_fourier_alignment_offset(image_channel1, image_channel3, file_name, preprocessing, firstpair=False)


        ShowOverlapRegion_3channels(image_channel1, image_channel2, image_channel3, best_offset2, best_offset3, file_name, scale_percent=300,output=show)
    else:
        if file_index == -1:
            file_name = random.choice(high_resolution_name_list)
        else:
            file_name = high_resolution_name_list[file_index]
        image_channel1, image_channel2, image_channel3 = ReadImage(file_name,'ps')
        print('[Info]Choose High resolution file: ', file_name)

        best_offset2 = compute_fourier_alignment_offset(image_channel1, image_channel2, file_name, preprocessing, firstpair=True)
        best_offset3 = compute_fourier_alignment_offset(image_channel1, image_channel3, file_name, preprocessing, firstpair=False)


        ShowOverlapRegion_3channels(image_channel1, image_channel2, image_channel3, best_offset2, best_offset3,file_name,scale_percent=30,output=show)
    

    end_time = time.time()
    total_time = end_time - start_time 
    print('[Info]Total time: ',total_time)
    return

if __name__ == "__main__":
    main()

