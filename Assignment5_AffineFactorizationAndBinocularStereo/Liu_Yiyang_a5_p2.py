# import required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

# # read two input images as grayscale images
# imgL = cv2.imread('stereo_data/room1.png',0)
# imgR = cv2.imread('stereo_data/room2.png',0)

# # Initiate and StereoBM object
# stereo = cv2.StereoBM_create(numDisparities=144, blockSize=31)

# # compute the disparity map
# disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity,'gray')
# plt.show()
# disparity.shape
def normalized_cross_correlation(block1, block2):
    # Compute mean of the blocks
    mean1 = np.mean(block1)
    mean2 = np.mean(block2)

    # Compute cross-correlation terms
    cross_corr = np.sum((block1 - mean1) * (block2 - mean2))
    std1 = np.sqrt(np.sum((block1 - mean1) ** 2))
    std2 = np.sqrt(np.sum((block2 - mean2) ** 2))

    # Compute normalized cross-correlation
    if std1 == 0 or std2 == 0:
        return 0  # to avoid division by zero
    else:
        return cross_corr / (std1 * std2)

def block_matching_stereo(imgL, imgR, block_size=5, num_disparities=16, matching_function='ssd'):
    h, w = imgL.shape

    # Initialize the disparity map with zeros
    disparity = np.zeros_like(imgL, dtype=np.float32)

    # Iterate over each pixel in the left image
    for y in range(block_size // 2, h - block_size // 2):
        for x in range(block_size // 2, w - block_size // 2):
            # Define the search range in the right image
            search_range = min(num_disparities, x)

            # Extract the block in the left image
            block_left = imgL[y - block_size // 2:y + block_size // 2 + 1, x - block_size // 2:x + block_size // 2 + 1]

            best_match_x = x
            min_ssd = float('inf')

            cost_list = []
            # Search for the best match in the right image
            for d in range(search_range):
                if x - d - block_size // 2 >= 0 and x - d + block_size // 2 + 1 <= w - block_size // 2:
                    # Extract the block in the right image
                    block_right = imgR[y - block_size // 2:y + block_size // 2 + 1, x - d - block_size // 2:x - d + block_size // 2 + 1]

                    # Compute the sum of squared differences (SSD)
                    if matching_function == 'ssd':
                        cost = np.sum((block_left - block_right) ** 2)
                    elif matching_function == 'sad':
                        cost = np.sum(np.abs(block_left - block_right))
                    elif matching_function == 'nc':
                        # cost = np.sum((block_left - np.mean(block_left)) * (block_right - np.mean(block_right))) / (
                        #         np.std(block_left) * np.std(block_right) * block_size ** 2)
                        cost = -np.sum((block_left - np.mean(block_left)) * (block_right - np.mean(block_right))) / (
                                np.std(block_left) * np.std(block_right) * block_size ** 2)
                        # cost = normalized_cross_correlation(block_left, block_right)
                    cost_list.append(cost)
                

                    # Update the best match if a smaller SSD is found
                    if cost < min_ssd:
                        min_ssd = cost
                        best_match_x = x - d
            # plt.figure(f"Matching Cost x {x} y {y}")
            # plt.plot(range(len(cost_list)), cost_list, marker='o')
            # plt.title(f"Matching Cost x {x} y {y}")
            # plt.xlabel('Disparity')
            # plt.ylabel('Cost')
            # plt.show()
            # Assign the disparity value to the corresponding pixel
            disparity[y, x] = x - best_match_x

    return disparity.astype(np.uint8)

num_disparities = 16  
method = 'nc'
# block_size_list = [5,7,9,10,11,13,15,17,19,21]
block_size_list = [15]
block_size_list = []


for block_size in block_size_list:
# block_size = 15
# num_disparities_list = [16,32,48,64,80]
# for num_disparities in num_disparities_list:
  
    
    # Read two input images as grayscale images
    imgL = cv2.imread('stereo_data/tsukuba1.jpg', 0)
    imgR = cv2.imread('stereo_data/tsukuba2.jpg', 0)
    imgL = imgL.astype(np.float32) / 255.0
    imgR = imgR.astype(np.float32) / 255.0

    start_time = time.time()
    # Set the block size and number of disparities
    # block_size = 11
    # num_disparities = 25

    # Compute the disparity map using block matching
    repeat = 1
    for i in range(repeat):
        disparity_map = block_matching_stereo(imgL, imgR, block_size=block_size, 
                                              num_disparities=num_disparities, matching_function='ssd')
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time /= repeat
    # Display the disparity map
    plt.figure(f"tsukuba block size {block_size} disparities {num_disparities} method {method}")
    # plt.imshow(disparity_map, cmap='gray')
    plt.imsave(f"tsukuba block size {block_size} disparities {num_disparities} method {method}.jpg",disparity_map, cmap='gray')
    # plt.show()
    print("disparity_map shape:", disparity_map.shape)


    print(f"time for tsukuba block size {block_size} disparities {num_disparities} method {method}: {elapsed_time} s")




# start_time = time.time()
# # Read two input images as grayscale images
# imgL = cv2.imread('stereo_data/moebius1.png', 0)
# imgR = cv2.imread('stereo_data/moebius2.png', 0)

# # Set the block size and number of disparities
# block_size = 15
# num_disparities = 80

# # Compute the disparity map using block matching
# disparity_map = block_matching_stereo(imgL, imgR, block_size=block_size, num_disparities=num_disparities, matching_function='ssd')
# end_time = time.time()

# # Display the disparity map
# plt.figure("moebius")
# plt.imshow(disparity_map, cmap='gray')
# plt.show()
# print("disparity_map shape:", disparity_map.shape)
# elapsed_time = end_time - start_time
# print(f"time for moebius: {elapsed_time} s")

# start_time = time.time()
# # Read two input images as grayscale images
# imgL = cv2.imread('stereo_data/room1.png', 0)
# imgR = cv2.imread('stereo_data/room2.png', 0)

# # Set the block size and number of disparities
# block_size = 31
# num_disparities = 144

# # Compute the disparity map using block matching
# disparity_map = block_matching_stereo(imgL, imgR, block_size=block_size, num_disparities=num_disparities, matching_function='ssd')
# end_time = time.time()

# # Display the disparity map
# plt.figure("room")
# plt.imshow(disparity_map, cmap='gray')
# plt.imsave(f"room block size {block_size} disparities {num_disparities} method ssd.jpg",disparity_map, cmap='gray')
# plt.show()
# print("disparity_map shape:", disparity_map.shape)
# elapsed_time = end_time - start_time
# print(f"time for room: {elapsed_time} s")


def disparity_to_depth(disparity_map, baseline, focal_length):
    # Convert disparity map to depth map
    depth_map = np.zeros_like(disparity_map, dtype=np.float32)
    non_zero_indices = disparity_map != 0
    depth_map[non_zero_indices] = focal_length * baseline / disparity_map[non_zero_indices]

    return depth_map

def visualize_3d_depth(depth_map, imgL, imgR):
    # Create 3D coordinates
    h, w = depth_map.shape
    y_coords, x_coords = np.mgrid[:h, :w]
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = depth_map.flatten()

    # Filter out points with zero depth
    non_zero_indices = z_coords != 0
    x_coords = x_coords[non_zero_indices]
    y_coords = y_coords[non_zero_indices]
    z_coords = z_coords[non_zero_indices]

    # Extract color information from the original images
    colors = imgL.flatten()
    # colors = np.column_stack(colors).astype(float) / 255.0

    # Filter out color values for points with zero depth
    colors = colors[non_zero_indices]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, c=colors, marker='s', s=1)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_zlim((0,30))
    plt.show()

def delaunay_triangulation(depth_map):
    # Create 3D coordinates
    h, w = depth_map.shape
    y_coords, x_coords = np.mgrid[:h, :w]
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    z_coords = depth_map.flatten()

    # Filter out points with zero depth
    non_zero_indices = z_coords != 0
    x_coords = x_coords[non_zero_indices]
    y_coords = y_coords[non_zero_indices]
    z_coords = z_coords[non_zero_indices]

    # Perform Delaunay triangulation
    points = np.column_stack((x_coords, y_coords, z_coords))
    tri = Delaunay(points)

    # Plot the triangulation
    plt.figure()
    plt.triplot(x_coords, y_coords, tri.simplices)
    plt.plot(x_coords, y_coords, 'o')

    # Set axis labels
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()

# Assuming baseline and focal length values (you may need to adjust these values)
baseline = 1.0
focal_length = 100.0  # Adjust this value based on your specific camera parameters

# Convert disparity map to depth map
# depth_map = disparity_to_depth(disparity_map, baseline, focal_length)

# # Visualize the depth map in 3D
# visualize_3d_depth(depth_map, imgL, imgR)

# # Perform Delaunay triangulation and visualize
# delaunay_triangulation(depth_map)

# ------------------------------------------------------------
# CALCULATE DISPARITY (DEPTH MAP)
# Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
# and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

# StereoSGBM Parameter explanations:
# https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html
# imgL = cv2.imread('stereo_data/tsukubal.jpg', cv2.IMREAD_GRAYSCALE)
# imgR = cv2.imread('stereo_data/tsukuba2.jpg', cv2.IMREAD_GRAYSCALE)
# Initiate and StereoBM object
# read two input images as grayscale images
imgL = cv2.imread('stereo_data/tsukuba1.jpg', 0)
imgR = cv2.imread('stereo_data/tsukuba2.jpg', 0)

# plt.figure("disparity map opencv")
# plt.imshow(disparity, cmap='plasma')
# plt.colorbar()
# plt.show()

# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 5
min_disp = 0
max_disp = 16
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 5
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 50
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
# Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
disp12MaxDiff = 9

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
    # mode=cv2.StereoSGBM_MODE_HH
)
disparity_SGBM = stereo.compute(imgL, imgR)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)

# cv2.imshow("Disparity", disparity_SGBM)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(f"SGBM tsukuba block size {block_size} disparities {max_disp}.jpg", disparity_SGBM)

# if True:
#     plt.imshow(disparity_SGBM, cmap='plasma')
#     plt.colorbar()
#     plt.show()

imgL = cv2.imread('stereo_data/moebius1.png', 0)
imgR = cv2.imread('stereo_data/moebius2.png', 0)

# plt.figure("disparity map opencv")
# plt.imshow(disparity, cmap='plasma')
# plt.colorbar()
# plt.show()

# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 3
min_disp = 0
max_disp = 64
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 5
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 50
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
# Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
disp12MaxDiff = 20

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
    # mode=cv2.StereoSGBM_MODE_HH
)
disparity_SGBM = stereo.compute(imgL, imgR)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)

# cv2.imshow("Disparity", disparity_SGBM)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(f"SGBM moebius block size {block_size} disparities {max_disp}.jpg", disparity_SGBM)