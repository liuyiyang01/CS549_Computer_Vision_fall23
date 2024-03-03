# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
## Fundamental Matrix Estimation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
import cv2
import random
from tqdm.notebook import tqdm
from matching_example import read_image, SIFT, plot_sift, matcher, plot_matches
# download file matching_example from https://gist.github.com/tigercosmos/90a5664a3b698dc9a4c72bc0fcbd21f4


print("----------------------------- 1. Fundamental matrix estimation from ground truth matches ------------------------------")
def fit_fundamental(matches):
    # Normalize the points (mean is 0, standard deviation is sqrt(2))
    # print("matches: ", matches)
    normalized_matches = matches.copy()
    # print("np.mean(matches[:, :2], axis=0):", np.mean(matches[:, :2], axis=0))
    normalized_matches[:, :2] -= np.mean(matches[:, :2], axis=0)
    normalized_matches[:, :2] /= np.std(matches[:, :2], axis=0)
    
    normalized_matches[:, 2:] -= np.mean(matches[:, 2:], axis=0)
    normalized_matches[:, 2:] /= np.std(matches[:, 2:], axis=0)

    # Build the design matrix A for t/he normalized algorithm
    A = np.zeros((len(matches), 9))
    A[:, 0] = normalized_matches[:, 0] * normalized_matches[:, 2]
    A[:, 1] = normalized_matches[:, 1] * normalized_matches[:, 2]

    A[:, 2] = normalized_matches[:, 2]
    A[:, 3] = normalized_matches[:, 0] * normalized_matches[:, 3]
    A[:, 4] = normalized_matches[:, 1] * normalized_matches[:, 3]
    A[:, 5] = normalized_matches[:, 3]
    A[:, 6:8] = normalized_matches[:, :2]
    A[:, 8] = 1


    # Solve for the fundamental matrix using homogeneous least squares
    _, _, V = np.linalg.svd(A)
    
    F = V[-1, :]
    # print("F vector: ", F)
    F = F.reshape((3, 3))
    # print("F: ", F)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    # print("U: ", U)
    # print("S: ", S)
    # print("Vt: ", Vt)
    S[2] = 0
    F = U.dot(np.diag(S)).dot(Vt)

    # Denormalize the fundamental matrix
    T1 = np.array([[1/np.std(matches[:, 0]), 0, -np.mean(matches[:, 0]) / np.std(matches[:, 0])],
                   [0, 1/np.std(matches[:, 1]), -np.mean(matches[:, 1]) / np.std(matches[:, 1])],
                   [0, 0, 1]])

    T2 = np.array([[1/np.std(matches[:, 2]), 0, -np.mean(matches[:, 2]) / np.std(matches[:, 2])],
                   [0, 1/np.std(matches[:, 3]), -np.mean(matches[:, 3]) / np.std(matches[:, 3])],
                   [0, 0, 1]])

    F = np.dot(np.dot(T2.T, F), T1)
    print("Fundamental Matrix: ", F)
    return F

def calculate_fundamental_matrix_residual(matches, F):
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
    residuals = np.abs(pt_line_dist)
    print("Residual: ", np.mean(residuals))


# Display two images side-by-side with matches
def display_matches(I1, I2, matches):
    I3 = np.zeros((I1.size[1],I1.size[0]*2,3) )
    I3[:,:I1.size[0],:] = I1
    I3[:,I1.size[0]:,:] = I2
    # print("I3 shape: ", I3.shape)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I3).astype(int))

    ax.plot(matches[:,0],matches[:,1],  '+r')
    ax.plot( matches[:,2]+I1.size[0],matches[:,3], '+r')
    ax.plot([matches[:,0], matches[:,2]+I1.size[0]],[matches[:,1], matches[:,3]], 'r')
    plt.show()

# Display second image with epipolar lines reprojected from the first image
def display_epipolar_lines(I2, matches, F):
    N = len(matches)
    M = np.c_[matches[:,0:2], np.ones((N,1))].transpose()
    L1 = np.matmul(F, M).transpose() # transform points from 
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:,0]**2 + L1[:,1]**2)
    L = np.divide(L1,np.kron(np.ones((3,1)),l).transpose())# rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:,2:4], np.ones((N,1))]).sum(axis = 1)
    closest_pt = matches[:,2:4] - np.multiply(L[:,0:2],np.kron(np.ones((2,1)), pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - np.c_[L[:,1], -L[:,0]]*10# offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:,1], -L[:,0]]*10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(I2).astype(int))
    ax.plot(matches[:,2],matches[:,3],  '+r')
    ax.plot([matches[:,2], closest_pt[:,0]],[matches[:,3], closest_pt[:,1]], 'r')
    ax.plot([pt1[:,0], pt2[:,0]],[pt1[:,1], pt2[:,1]], 'g')
    plt.show()
##
## load images and match files for the first example
##

I1 = Image.open('MP4_part2_data/library1.jpg')
I2 = Image.open('MP4_part2_data/library2.jpg')
matches = np.loadtxt('MP4_part2_data/library_matches.txt')

I1 = Image.open('MP4_part2_data/lab1.jpg')
I2 = Image.open('MP4_part2_data/lab2.jpg')
matches = np.loadtxt('MP4_part2_data/lab_matches.txt')

# Display matches
display_matches(I1, I2, matches)

# Fit fundamental matrix and display epipolar lines
F = fit_fundamental(matches)
display_epipolar_lines(I2, matches, F)
calculate_fundamental_matrix_residual(matches, F)

# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image: 
# matches(i,1:2) is a point in the first image
# matches(i,3:4) is a corresponding point in the second image



##
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
##





## Camera Calibration
print("----------------------------- 2. Camera calibration ------------------------------")

def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

def camera_calibration(matches_file, points_3d_file):
    # Load matches and 3D points
    matches = np.loadtxt(matches_file)
    points_3d_2 = np.loadtxt(points_3d_file)
    # print(points_3d_2)
    N = len(matches)
    points_3d = np.c_[points_3d_2, np.ones((N,1))]
    # print("points_3d: ", points_3d)
    # Construct the A matrix for the linear system
    A = np.zeros((2 * N, 12))
    for i in range(N):
        A[2 * i, :4] = points_3d[i]
        A[2 * i, 8:12] = -matches[i, 0]*points_3d[i]
        A[2 * i + 1, 4:8] = points_3d[i]
        A[2 * i + 1, 8:12] = -matches[i, 1]*points_3d[i]
    # print("A: ", A)
    # Solve the linear system to get the camera projection matrix
    _, _, V = np.linalg.svd(A)
    P1 = V[-1, :12].reshape((3, 4))

    # Normalize to make the last element of the last row equal to 1
    P1 /= P1[-1, -1]

    # Evaluate the camera matrix
    points_3d_proj, residual = evaluate_points(P1, matches[:, :2], np.loadtxt(points_3d_file))

    # Display the results
    print("Lab Data Camera Projection Matrix (Camera 1):\n", P1)
    print("Lab Data Calibration Residual Error:", residual)
    
    A = np.zeros((2 * N, 12))
    for i in range(N):
        A[2 * i, :4] = points_3d[i]
        A[2 * i, 8:12] = -matches[i, 2]*points_3d[i]
        A[2 * i + 1, 4:8] = points_3d[i]
        A[2 * i + 1, 8:12] = -matches[i, 3]*points_3d[i]
    # print("A: ", A)
    # Solve the linear system to get the camera projection matrix
    _, _, V = np.linalg.svd(A)
    P2 = V[-1, :12].reshape((3, 4))

    # Normalize to make the last element of the last row equal to 1
    P2 /= P2[-1, -1]

    # Evaluate the camera matrix
    points_3d_proj, residual = evaluate_points(P2, matches[:, 2:], np.loadtxt(points_3d_file))

    # Display the results
    print("Lab Data Camera Projection Matrix (Camera 2):\n", P2)
    print("Lab Data Calibration Residual Error:", residual)

    return P1, P2, points_3d_proj

# Camera calibration for the lab pair
lab_matches_file = 'MP4_part2_data/lab_matches.txt'
lab_points_3d_file = 'MP4_part2_data/lab_3d.txt'
# image_size_lab = I1.size[::-1]  # Reverse the size for proper image dimensions

P_lab, P_lab2, points_3d_proj = camera_calibration(lab_matches_file, lab_points_3d_file)





## Camera Centers
print("----------------------------- 3. Calculate the camera centers ------------------------------")

def calculate_camera_center(projection_matrix):
    # Perform SVD on the projection matrix
    _, _, Vt = np.linalg.svd(projection_matrix)

    # Extract the last column of Vt to get the camera center
    camera_center = Vt[-1, :]

    # Normalize the camera center (divide by the last element)
    camera_center /= camera_center[-1]

    return camera_center[:-1]  # Exclude the last element (homogeneous coordinate)



camera_center_lab = calculate_camera_center(P_lab)
camera_center_lab2 = calculate_camera_center(P_lab2)
print("Camera Center for the Lab Pair Camera 1:", camera_center_lab)
print("Camera Center for the Lab Pair Camera 2:", camera_center_lab2)

# Calculate camera centers for the library pair
P_library1 = np.loadtxt('MP4_part2_data/library1_camera.txt')
P_library2 = np.loadtxt('MP4_part2_data/library2_camera.txt')
camera_center_library1 = calculate_camera_center(P_library1)
camera_center_library2 = calculate_camera_center(P_library2)
print("Camera Center for Library Camera 1:", camera_center_library1)
print("Camera Center for Library Camera 2:", camera_center_library2)






## Triangulation
print("----------------------------- 4. Triangulation ------------------------------")

def triangulate(P1, P2, matches):
    N = len(matches)
    X = np.zeros((N, 3))

    for i in range(N):
        A = np.vstack((matches[i, 0] * P1[2, :] - P1[0, :],
                       matches[i, 1] * P1[2, :] - P1[1, :],
                       matches[i, 2] * P2[2, :] - P2[0, :],
                       matches[i, 3] * P2[2, :] - P2[1, :]))

        _, _, V = np.linalg.svd(A)
        X[i, :] = V[-1, :3] / V[-1, -1]
    # print(X)
    return X


def calculate_residual(matches, points_3d, P1, P2):
    # Project 3D points to 2D using the projection matrices
    points_2d_proj1, _ = evaluate_points(P1, matches[:, :2], points_3d)
    points_2d_proj2, _ = evaluate_points(P2, matches[:, 2:4], points_3d)

    # Calculate residuals
    residual1 = np.sqrt(np.sum((points_2d_proj1 - matches[:, :2])**2, axis=1))
    residual2 = np.sqrt(np.sum((points_2d_proj2 - matches[:, 2:4])**2, axis=1))

    return residual1, residual2

# Triangulation for the lab pair
matches = np.loadtxt('MP4_part2_data/lab_matches.txt')
points_3d_lab = triangulate(P_lab, P_lab2, matches)
points_3d_lab_truth = np.loadtxt(lab_points_3d_file)
# Calculate residuals for the lab pair
residual_lab1, residual_lab2 = calculate_residual(matches, points_3d_lab, P_lab, P_lab2)

# Display the camera centers and reconstructed points in 3D for the lab pair
fig = plt.figure("Triangulation for the lab pair")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(camera_center_lab[0], camera_center_lab[1], camera_center_lab[2], c='r', marker='o', label='Camera Center Lab')
ax.scatter(camera_center_lab2[0], camera_center_lab2[1], camera_center_lab2[2], c='g', marker='o', label='Camera Center Lab Camera 2')

ax.scatter(points_3d_lab[:, 0], points_3d_lab[:, 1], points_3d_lab[:, 2], c='b', marker='.', label='Reconstructed Points Lab')
ax.scatter(points_3d_lab_truth[:, 0], points_3d_lab_truth[:, 1], points_3d_lab_truth[:, 2], c='y', marker='.', label='Ground Truth Points Lab')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

# Display residuals for the lab pair
print("Residuals for Lab Pair:")
print("Mean Residual for Image 1:", np.mean(residual_lab1))
print("Mean Residual for Image 2:", np.mean(residual_lab2))

# Triangulation for the library pair
matches = np.loadtxt('MP4_part2_data/library_matches.txt')
points_3d_library = triangulate(P_library1, P_library2, matches)

# Calculate residuals for the library pair
residual_library1, residual_library2 = calculate_residual(matches, points_3d_library, P_library1, P_library2)

# Display the camera centers and reconstructed points in 3D for the library pair
fig = plt.figure("Triangulation for the library pair")
ax = fig.add_subplot(111, projection='3d')
ax.scatter(camera_center_library1[0], camera_center_library1[1], camera_center_library1[2], c='r', marker='o', label='Camera Center Library 1')
ax.scatter(camera_center_library2[0], camera_center_library2[1], camera_center_library2[2], c='g', marker='o', label='Camera Center Library 2')
ax.scatter(points_3d_library[:, 0], points_3d_library[:, 1], points_3d_library[:, 2], c='b', marker='.', label='Reconstructed Points Library')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

# Display residuals for the library pair
print("\nResiduals for Library Pair:")
print("Mean Residual for Image 1:", np.mean(residual_library1))
print("Mean Residual for Image 2:", np.mean(residual_library2))



##Fundamental matrix estimation without ground-truth matches
print("----------------------------- 5. Fundamental matrix estimation without ground-truth matches ------------------------------")


# plt.rcParams['figure.figsize'] = [15, 15]

# left_gray, left_origin, left_rgb = read_image('MP4_part2_data/house1.jpg')
# right_gray, right_origin, right_rgb = read_image('MP4_part2_data/house2.jpg')

left_gray, left_origin, left_rgb = read_image('MP4_part2_data/gaudi1.jpg')
right_gray, right_origin, right_rgb = read_image('MP4_part2_data/gaudi2.jpg')

# Better result when using gray
kp_left, des_left = SIFT(left_gray)
kp_right, des_right = SIFT(right_gray)

# kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
# kp_right_img = plot_sift(right_gray, right_rgb, kp_right)
# total_kp = np.concatenate((kp_left_img, kp_right_img), axis=1)
# plt.imshow(total_kp)
# I1 = Image.open('MP4_part2_data/house1.jpg')
# I2 = Image.open('MP4_part2_data/house2.jpg')
I1 = Image.open('MP4_part2_data/gaudi1.jpg')
I2 = Image.open('MP4_part2_data/gaudi2.jpg')
matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)

# total_img = np.concatenate((left_rgb, right_rgb), axis=1)
# plot_matches(matches, total_img) # Good mathces


# Display matches
display_matches(I1, I2, matches)
print("# of inliers: ", len(matches))
# Fit fundamental matrix and display epipolar lines
F = fit_fundamental(matches)
display_epipolar_lines(I2, matches, F)
calculate_fundamental_matrix_residual(matches, F)