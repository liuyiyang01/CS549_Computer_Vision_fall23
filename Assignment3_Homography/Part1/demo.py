import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# imports
import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy 

def imshowImage(image, scale_percent=100):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv.resize(image, (width, height))
    cv.imshow('image',resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


MIN_MATCH_COUNT = 50
# Load both images
image1 = cv.imread('data/right.jpg')
image2 = cv.imread('data/left.jpg')
# Convert images to double and grayscale
img1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

# img1 = cv.imread('data/left.jpg', cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('data/right.jpg', cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# Compute descriptor distances using Euclidean distance
descriptor_distances = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')

# Set a threshold for matching descriptors
descriptor_threshold = 15000  # You can adjust this value

# Find the indices of matches below the threshold
matches = np.where(descriptor_distances < descriptor_threshold)
print(matches)
good = []
for i in range(matches[0].shape[0]):
    queryIdx = matches[0][i]  # Index in the first set of descriptors
    trainIdx = matches[1][i]  # Index in the second set of descriptors
    d = descriptor_distances[queryIdx, trainIdx]  # Descriptor distance (you can use this as a matching score)
    match = cv2.DMatch(queryIdx, trainIdx, 0, d)
    good.append(match)

# # compute descriptor distance
# bf=cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# #matches = bf.match(descs0,descs1)
# # ratio test
# mkpts0=[]
# mkpts1=[]
# good =[] 
# for m, n in matches:
#     if m.distance < 0.5 * n.distance: 
#         good.append([m])


# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 50)
# search_params = dict(checks = 1000)
# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)
print("good match number: ", len(good))

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,1.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    # Calculate the number of homography inliers
    # print(src_pts)
    num_inliers = np.sum(matchesMask)
    # print(matchesMask)
    # Calculate the residuals for inliers
    inlier_residuals = []
    transformed_points = cv.perspectiveTransform(src_pts,M)
    # print(transformed_points)
    for i in range(len(matchesMask)):
        if matchesMask[i]:
            src_point = src_pts[i][0]
            dst_point = src_pts[i][0]
            transformed_point = transformed_points[i][0]
            residual = np.linalg.norm(transformed_point - dst_point)
            inlier_residuals.append(residual)

    # Calculate the average residual for inliers
    average_residual = np.mean(inlier_residuals)
    print("Number of Homography Inliers:", num_inliers)
    print("Average Residual for Inliers:", average_residual)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()

# Calculate the average residual for the inliers
if M is not None:
    print(M)
    print((image1.shape[1] + image2.shape[1], image2.shape[0]))
    warped_image2 = cv.warpPerspective(image1, M, (image1.shape[1] + image2.shape[1], image2.shape[0]),cv.INTER_LINEAR)
    
    # Blend the two images together by taking the maximum pixel values

    warped_image2[:image2.shape[0],:image2.shape[1]]=image2
    # imshowImage(warped_image2)
    cv2.imwrite("data/merged_image.jpg", warped_image2)
else:
    panorama = image1


# # Create a new image to hold the panorama
# result_image = np.zeros((panorama.shape[0], panorama.shape[1], 3), dtype=np.uint8)

# # Copy the images onto the result image
# result_image[:image1.shape[0], :image1.shape[1]] = image1
# result_image[:warped_image2.shape[0], image1.shape[1]:] = warped_image2

# # Display the final panorama
# plt.imshow(cv.cvtColor(result_image, cv.COLOR_BGR2RGB))
# plt.show()
