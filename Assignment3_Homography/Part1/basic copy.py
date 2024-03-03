# imports
import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy 


##############################################
### Provided code - nothing to change here ###
##############################################

"""
Harris Corner Detector
Usage: Call the function harris(filename) for corner detection
Reference   (Code adapted from):
             http://www.kaij.org/blog/?p=89
             Kai Jiang - Harris Corner Detector in Python
             
"""
from pylab import *
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image

def harris(filename, min_distance = 10, threshold = 0.1):
    """
    filename: Path of image file
    threshold: (optional)Threshold for corner detection
    min_distance : (optional)Minimum number of pixels separating 
     corners and image boundary
    """
    im = np.array(Image.open(filename).convert("L"))
    harrisim = compute_harris_response(im)
    filtered_coords = get_harris_points(harrisim,min_distance, threshold)
    plot_harris_points(im, filtered_coords)

def gauss_derivative_kernels(size, sizey=None):
    """ returns x and y derivatives of a 2D 
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = mgrid[-size:size+1, -sizey:sizey+1]
    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = - x * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    gy = - y * exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 
    return gx,gy

def gauss_kernel(size, sizey = None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = mgrid[-size:size+1, -sizey:sizey+1]
    g = exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()

def compute_harris_response(im):
    """ compute the Harris corner detector response function 
        for each pixel in the image"""
    #derivatives
    gx,gy = gauss_derivative_kernels(3)
    imx = signal.convolve(im,gx, mode='same')
    imy = signal.convolve(im,gy, mode='same')
    #kernel for blurring
    gauss = gauss_kernel(3)
    #compute components of the structure tensor
    Wxx = signal.convolve(imx*imx,gauss, mode='same')
    Wxy = signal.convolve(imx*imy,gauss, mode='same')
    Wyy = signal.convolve(imy*imy,gauss, mode='same')   
    #determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy   
    return Wdet / Wtr

def get_harris_points(harrisim, min_distance=10, threshold=0.1):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating 
        corners and image boundary"""
    #find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1    
    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]    
    #sort candidates
    index = argsort(candidate_values)   
    #store allowed point locations in array
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1   
    #select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),
                (coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0               
    return filtered_coords

def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'r*')
    axis('off')
    show()

# Usage: 
#harris('./path/to/image.jpg')


# Provided code for plotting inlier matches between two images

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')
    
# Usage:
# fig, ax = plt.subplots(figsize=(20,10))
# plot_inlier_matches(ax, img1, img2, computed_inliers)


#######################################
### Your implementation starts here ###
#######################################

def imshowImage(image, scale_percent=100):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))
    cv2.imshow('image',resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Load both images
image1 = cv2.imread('data/left.jpg')
image2 = cv2.imread('data/right.jpg')

# Convert images to double and grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.double)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.double)

# Detect feature points in both images
harris_image1 = compute_harris_response(gray_image1)
harris_image2 = compute_harris_response(gray_image2)

# Set your own threshold and minimum distance values
threshold = 0.01
min_distance = 10

# Get filtered corner coordinates for both images
filtered_coords1 = get_harris_points(harris_image1, min_distance, threshold)
filtered_coords2 = get_harris_points(harris_image2, min_distance, threshold)

# Convert images to double and grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.uint8)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Compute SIFT descriptors and keypoints for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)
# Convert keypoints to NumPy arrays
keypoints1_array = np.array([kp.pt for kp in keypoints1])
keypoints2_array = np.array([kp.pt for kp in keypoints2])

# Compute descriptor distances using Euclidean distance
descriptor_distances = scipy.spatial.distance.cdist(descriptors1, descriptors2, 'sqeuclidean')

# Set a threshold for matching descriptors
descriptor_threshold = 30000  # You can adjust this value

# Find the indices of matches below the threshold
matches = np.where(descriptor_distances < descriptor_threshold)
print(matches)

# Implement RANSAC to estimate a homography
from skimage.measure import ransac

# Define the minimum number of matches to consider
min_matches = 20  # You can adjust this value

# Initialize variables to keep track of the best transformation
best_inliers = []
best_homography = None

# RANSAC parameters
ransac_iterations = 100  # You can adjust this value
ransac_threshold = 5  # You can adjust this value

for i in range(ransac_iterations):
    # print("iteration: ", i)
    # Randomly select a subset of matches
    random_indices = np.random.choice(len(matches[0]), min_matches, replace=False)
    random_matches = (matches[0][random_indices], matches[1][random_indices])


    # Estimate the homography using the random matches
    homography, inliers = cv2.findHomography(keypoints1_array[random_matches[0]], keypoints2_array[random_matches[1]], cv2.RANSAC, ransac_threshold)
    # print(inliers)
    print(sum(inliers))
    # Check if this transformation has more inliers
    if sum(inliers) > sum(best_inliers):
        best_inliers = inliers
        best_homography = homography

# Calculate the average residual for the inliers
if best_homography is not None:
    inlier_residuals = np.linalg.norm(keypoints1_array[best_inliers] - cv2.perspectiveTransform(keypoints2_array[best_inliers], best_homography), axis=2)
    average_residual = np.mean(inlier_residuals)
    num_inliers = len(best_inliers)
else:
    # No valid transformation found
    average_residual = None
    num_inliers = 0

if best_homography is not None:
    # Warp the second image onto the first image
    warped_image2 = cv2.warpPerspective(image2, best_homography, (image1.shape[1] + image2.shape[1], image2.shape[0]))
    imshowImage(warped_image2)
    # Blend the two images together by taking the maximum pixel values
    panorama = np.maximum(image1, warped_image2)
else:
    # No valid transformation found, use the first image as the panorama
    panorama = image1


# Create a new image to hold the panorama
result_image = np.zeros((panorama.shape[0], panorama.shape[1], 3), dtype=np.uint8)

# Copy the images onto the result image
result_image[:image1.shape[0], :image1.shape[1]] = image1
result_image[:warped_image2.shape[0], image1.shape[1]:] = warped_image2

# Display the final panorama
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.show()



# # Create a list of DMatch objects for inliers
# inlier_matches = [cv2.DMatch(i, i, 0) for i in range(len(inliers))]

# # Draw the inlier matches between the two images
# matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, inlier_matches, outImg=None, matchColor=(0, 255, 0), singlePointColor=None)

# # Display the matched image
# plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
# plt.show()