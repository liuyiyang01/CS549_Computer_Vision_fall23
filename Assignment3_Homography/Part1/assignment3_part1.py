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
    return filtered_coords

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

def stitch_images(image1, image2):
    # Load and convert images to grayscale
    im1 = cv2.imread(image1, 0).astype(np.double)
    im2 = cv2.imread(image2, 0).astype(np.double)

    # Detect feature points using the Harris corner detector
    harrisim1 = compute_harris_response(im1)
    harrisim2 = compute_harris_response(im2)

    # Get the Harris points for both images
    filtered_coords1 = get_harris_points(harrisim1)
    filtered_coords2 = get_harris_points(harrisim2)

    # Extract local neighborhoods around keypoints
    neighborhood_size = 5  # Experiment with different sizes
    keypoints1 = [im1[y - neighborhood_size:y + neighborhood_size + 1, x - neighborhood_size:x + neighborhood_size + 1].ravel() for y, x in filtered_coords1]
    keypoints2 = [im2[y - neighborhood_size:y + neighborhood_size + 1, x - neighborhood_size:x + neighborhood_size + 1].ravel() for y, x in filtered_coords2]

    # Compute distances between descriptors using Euclidean distance
    descriptor_distances = scipy.spatial.distance.cdist(keypoints1, keypoints2, 'sqeuclidean')

    # Set a threshold for selecting putative matches
    threshold = 50000  # Experiment with different values
    putative_matches = np.argwhere(descriptor_distances < threshold)
    print(putative_matches)
    # Implement RANSAC to estimate a homography
    num_iterations = 100  # Experiment with different numbers
    inlier_threshold = 1  # Experiment with different values

    best_inliers = []
    best_homography = None

    for _ in range(num_iterations):
        # Randomly select 4 putative matches
        random_indices = np.random.choice(putative_matches.shape[0], 4, replace=False)
        random_matches = putative_matches[random_indices]

        # Estimate a homography
        A = []
        for match in random_matches:
            x1, y1 = filtered_coords1[match[0]]
            x2, y2 = filtered_coords2[match[1]]
            A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
            A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)

        # Calculate the number of inliers and the average residual
        inliers = []
        total_residual = 0
        for match in putative_matches:
            x1, y1 = filtered_coords1[match[0]]
            x2, y2 = filtered_coords2[match[1]]
            p1 = np.array([x1, y1, 1])
            p2 = np.dot(H, p1)
            p2 /= p2[2]
            distance = np.linalg.norm(p2[:2] - np.array([x2, y2]))
            if distance < inlier_threshold:
                inliers.append(match)
                total_residual += distance

        # Update the best homography if more inliers are found
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_homography = H
            print(best_inliers)

    # Warp one image onto the other using the estimated transformation
    warped_image = cv2.warpPerspective(im1, best_homography, (im1.shape[1] + im2.shape[1], im2.shape[0]))
    warped_image[:im2.shape[0], :im2.shape[1]] = im2

    return warped_image

# Load the images and stitch them
image1 = 'data/left.jpg'
image2 = 'data/right.jpg'

result = stitch_images(image1, image2)

# Display the stitched image
plt.figure(figsize=(12, 6))
plt.imshow(result, cmap='gray')
plt.axis('off')
plt.show()
