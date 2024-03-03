import numpy as np
import cv2
import random
import time


def ReadImage(file_name, method='pixel'):
    if method == 'evenly':
        #using crop by evenly dividing by 3
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

    # # show the image of each channel
    # imshowImage(image_channel1)
    # imshowImage(image_channel2)
    # imshowImage(image_channel3)

    return image_channel1, image_channel2, image_channel3

def ncc(image1, image2):
    # 转换图像为浮点数
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # 计算图像均值
    mean_image1 = np.mean(image1)
    mean_image2 = np.mean(image2)

    # 计算零均值图像
    zero_mean_image1 = image1 - mean_image1
    zero_mean_image2 = image2 - mean_image2

    # 计算分子部分：两个图像的点积
    numerator = np.sum(zero_mean_image1 * zero_mean_image2)

    # 计算分母部分：两个图像的欧几里得范数
    norm_image1 = np.sqrt(np.sum(zero_mean_image1 ** 2))
    norm_image2 = np.sqrt(np.sum(zero_mean_image2 ** 2))

    # 计算NCC值
    ncc_value = numerator / (norm_image1 * norm_image2)

    return ncc_value

def GaussianDownsampling(input_image,num_levels=8):
    # Create an empty list to store the pyramid layers
    gaussian_pyramid = [input_image]

    # # Define the number of pyramid levels
    # num_levels = 8

    # Build the Gaussian pyramid
    for i in range(1, num_levels):
        # Apply Gaussian blur to the previous level
        previous_level = gaussian_pyramid[i - 1]
        blurred = cv2.GaussianBlur(previous_level, (5, 5), 0)
        
        # Downsample the blurred image
        downsampled = cv2.pyrDown(blurred)
        
        # Add the downsampled image to the pyramid
        gaussian_pyramid.append(downsampled)

    # # Display or save the pyramid layers as needed
    # for i, layer in enumerate(gaussian_pyramid):
    #     cv2.imshow(f'Level {i}', layer)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return gaussian_pyramid

def align_images(image1, image2, metric, window_size=15, method='NORMAL', pdx=0, pdy=0):
    coarse_range = 2
    best_score = float('-inf')
    best_offset = (0, 0)


    h1, w1 = image1.shape[0:2]
    h2, w2 = image2.shape[0:2]

    if method == 'NORMAL':
        for dx in range(-window_size, window_size + 1):
            for dy in range(-window_size, window_size + 1):
                # shifted_image2 = np.roll(image2, (dy, dx), axis=(0, 1))

                # Calculate the overlap region
                x1 = max(0, dx)
                x2 = min(w1, w2 + dx)
                y1 = max(0, dy)
                y2 = min(h1, h2 + dy)


                if x1 < x2 and y1 < y2:
                    overlap_image1 = image1[y1:y2, x1:x2]
                    overlap_image2 = image2[y1 - dy:y2 - dy, x1 - dx:x2 - dx]
                    if metric == 'ncc':
                        score = ncc(overlap_image1, overlap_image2)
                    # elif metric == 'ssd':
                    #     score = ssd(overlap_image1, overlap_image2)


                    if score > best_score:
                        best_score = score
                        best_offset = (dx, dy)

                    # print(dx,dy,score,best_score)
    else:
        for dx in range(2*pdx-coarse_range, 2*pdx+coarse_range+1):
            for dy in range(2*pdy-coarse_range, 2*pdy+coarse_range+1):
                # shifted_image2 = np.roll(image2, (dy, dx), axis=(0, 1))

                # Calculate the overlap region
                x1 = max(0, dx)
                x2 = min(w1, w2 + dx)
                y1 = max(0, dy)
                y2 = min(h1, h2 + dy)


                if x1 < x2 and y1 < y2:
                    overlap_image1 = image1[y1:y2, x1:x2]
                    overlap_image2 = image2[y1 - dy:y2 - dy, x1 - dx:x2 - dx]
                    if metric == 'ncc':
                        score = ncc(overlap_image1, overlap_image2)

                    if score > best_score:
                        best_score = score
                        best_offset = (dx, dy)

                    # print(dx,dy,score,best_score)
    print('[Info]Best offset: ', best_offset)      
    return best_offset


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

def ShowOverlapRegion_3channels(image1, image2, image3, offset2, offset3,file_name,scale_percent,output='SHOW'):
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

    if output == 'SHOW':
        imshowImage(merged_image, scale_percent)

    return

def CoarseToFine(gaussian_pyramid_channel1,gaussian_pyramid_channel2):
    metric = 'ncc'
    window_size = 5
    

    for i in range(len(gaussian_pyramid_channel1)-1,-1,-1):
        if i == len(gaussian_pyramid_channel1)-1:
            best_offset = align_images(gaussian_pyramid_channel1[i], gaussian_pyramid_channel2[i], metric, window_size)
        else:
            best_offset = align_images(gaussian_pyramid_channel1[i], gaussian_pyramid_channel2[i], metric, method='COARSETOFINE', pdx=pdx, pdy=pdy)
        pdx = best_offset[0]
        pdy = best_offset[1]
        h1, w1 = gaussian_pyramid_channel1[i].shape[0:2]
        h2, w2 = gaussian_pyramid_channel2[i].shape[0:2]
        
    # # Calculate the overlap region
    # x1 = max(0, dx)
    # x2 = min(w1, w2 + dx)
    # y1 = max(0, dy)
    # y2 = min(h1, h2 + dy)
    print('[CoarseToFine]Best offset:',best_offset)
    return best_offset

def main():
    start_time = time.time()
    file_name_list = ['00125v', '00149v', '00153v', '00351v', '00398v', '01112v']
    high_resolution_name_list = ['01047u', '01657u', '01861a']
    file_type = 1 #set to 0 to process normal images, set to 1 for high resolution images
    file_index = 2 #set to -1 to let program to randomly choose a image
    
    num_levels = 8 #iteration of the image pyramid
    
    
    if not file_type:
        if file_index == -1:
            file_name = random.choice(file_name_list)
        else:
            file_name = file_name_list[file_index]
        image_channel1, image_channel2, image_channel3 = ReadImage(file_name,'pixel')
        print('[Info]Choose file: ', file_name)
        metric = 'ncc'
        window_size = 15

        best_offset2 = align_images(image_channel1, image_channel2, metric, window_size)
        best_offset3 = align_images(image_channel1, image_channel3, metric, window_size)


        ShowOverlapRegion_3channels(image_channel1, image_channel2, image_channel3, best_offset2, best_offset3, file_name, scale_percent=300,output='SHOW')
    else:
        if file_index == -1:
            file_name = random.choice(high_resolution_name_list)
        else:
            file_name = high_resolution_name_list[file_index]
        image_channel1, image_channel2, image_channel3 = ReadImage(file_name,'ps')
        print('[Info]Choose High resolution file: ', file_name)
        
        
        gaussian_pyramid_channel1 = GaussianDownsampling(image_channel1, num_levels)
        gaussian_pyramid_channel2 = GaussianDownsampling(image_channel2, num_levels)
        gaussian_pyramid_channel3 = GaussianDownsampling(image_channel3, num_levels)

        best_offset2 = CoarseToFine(gaussian_pyramid_channel1,gaussian_pyramid_channel2)
        best_offset3 = CoarseToFine(gaussian_pyramid_channel1,gaussian_pyramid_channel3)


        ShowOverlapRegion_3channels(image_channel1, image_channel2, image_channel3, best_offset2, best_offset3,file_name,scale_percent=30,output='SHOW')
    

    end_time = time.time()
    total_time = end_time - start_time 
    print('[Info]Total time: ',total_time)
    return

if __name__ == "__main__":
    main()




