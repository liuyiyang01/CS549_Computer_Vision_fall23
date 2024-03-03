#!/usr/bin/env python
# coding: utf-8

# # Part 3: Single-View Geometry
# 
# ## Usage
# This code snippet provides an overall code structure and some interactive plot interfaces for the *Single-View Geometry* section of Assignment 3. In [main function](#Main-function), we outline the required functionalities step by step. Some of the functions which involves interactive plots are already provided, but [the rest](#Your-implementation) are left for you to implement.
# 
# ## Package installation
# - In this code, we use `tkinter` package. Installation instruction can be found [here](https://anaconda.org/anaconda/tk).

# # Common imports

# In[364]:


get_ipython().run_line_magic('matplotlib', 'tk')
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from PIL import Image


# # Provided functions

# In[365]:


def get_input_lines(im, i, min_lines=3):
    """
    Allows user to input line segments; computes centers and directions.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        min_lines: minimum number of lines required
    Returns:
        n: number of lines from input
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        centers: np.ndarray of shape (3, n)
            where each column denotes the homogeneous coordinates of the centers
    """
    
    if(i==0):
        n = 7
        lines = np.array([
            [3.71612903e+01, -4.12903226e+01, 2.47741935e+01, 1.65161290e+01, 8.25806452e+00, -4.12903226e+00, -1.23870968e+01],
            [5.01677419e+02, -5.01677419e+02, 4.50064516e+02, 4.52129032e+02, 4.52129032e+02, 4.48000000e+02, 4.45935484e+02],
            [-9.73744483e+04, 9.56206085e+04, -8.95023151e+04, -9.08330889e+04, -9.45240441e+04, -9.59388504e+04, -9.90831484e+04]
        ])
        centers = np.array([
            [578.59677419, 578.59677419, 565.17741935, 568.27419355, 566.20967742, 566.20967742, 567.24193548],
            [151.23870968, 142.98064516, 167.75483871, 180.14193548, 198.72258065, 219.36774194, 237.9483871 ],
            [1, 1, 1, 1, 1, 1, 1]
        ])
        print("i = 0, use saved data")
    elif(i==1):
        n = 7  # Number of lines
        lines = np.array([
            [-1.75483871e+02, -1.11483871e+02, -8.67096774e+01, -1.03225806e+02, -1.28000000e+02, -1.30064516e+02, -7.22580645e+01],
            [-8.25806452e+00, -8.25806452e+00, -4.12903226e+00, 0.00000000e+00, 0.00000000e+00, -2.06451613e+00, 2.06451613e+00],
            [1.46579853e+05, 1.08200664e+05, 8.03955746e+04, 5.38655567e+04, 6.01868387e+04, 9.15611505e+04, 2.33453553e+04]
        ])

        centers = np.array([
            [825.30645161, 955.37096774, 918.20967742, 521.82258065, 470.20967742, 700.40322581, 328.79032258],
            [212.14193548, 204.91612903, 188.4, 202.8516129, 217.30322581, 224.52903226, 199.75483871],
            [1, 1, 1, 1, 1, 1, 1]
        ])
        print("i = 1, use saved data")
    elif(i==2):
        n = 5
        lines = np.array([
            [-2.68387097e+01, -1.44516129e+01, 1.03225806e+01, 6.19354839e+00, 0.00000000e+00],
            [1.25935484e+02, 1.05290323e+02, -1.07354839e+02, -1.05290323e+02, -1.11483871e+02],
            [6.91000874e+03, -4.19302560e+03, 1.00057307e+04, 1.54923222e+04, 2.51464458e+04]
        ])

        centers = np.array([
            [894.46774194, 894.46774194, 893.43548387, 894.46774194, 889.30645161],
            [135.75483871, 162.59354839, 179.10967742, 199.75483871, 225.56129032],
            [1, 1, 1, 1, 1]
        ])
        print("i = 2, use saved data")
    else:
        n = 0
        lines = np.zeros((3, 0))
        centers = np.zeros((3, 0))

        plt.figure()
        plt.imshow(im)
        plt.show()
        print('Set at least %d lines to compute vanishing point' % min_lines)
        while True:
            print('Click the two endpoints, use the right key to undo, and use the middle key to stop input')
            clicked = plt.ginput(2, timeout=0, show_clicks=True)
            if not clicked or len(clicked) < 2:
                if n < min_lines:
                    print('Need at least %d lines, you have %d now' % (min_lines, n))
                    continue
                else:
                    # Stop getting lines if number of lines is enough
                    break

            # Unpack user inputs and save as homogeneous coordinates
            pt1 = np.array([clicked[0][0], clicked[0][1], 1])
            pt2 = np.array([clicked[1][0], clicked[1][1], 1])
            # Get line equation using cross product
            # Line equation: line[0] * x + line[1] * y + line[2] = 0
            line = np.cross(pt1, pt2)
            lines = np.append(lines, line.reshape((3, 1)), axis=1)
            # Get center coordinate of the line segment
            center = (pt1 + pt2) / 2
            centers = np.append(centers, center.reshape((3, 1)), axis=1)

            # Plot line segment
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='b')

            n += 1
        print("n: ", n)
        print("lines: ", lines)
        print("center: ", centers)

    return n, lines, centers


# In[366]:


def plot_lines_and_vp(im, lines, vp):
    """
    Plots user-input lines and the calculated vanishing point.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        lines: np.ndarray of shape (3, n)
            where each column denotes the parameters of the line equation
        vp: np.ndarray of shape (3, )
    """
    bx1 = min(1, vp[0] / vp[2]) - 10
    bx2 = max(im.shape[1], vp[0] / vp[2]) + 10
    by1 = min(1, vp[1] / vp[2]) - 10
    by2 = max(im.shape[0], vp[1] / vp[2]) + 10
    print("Bounding Box Coordinates:")
    print("Top-left corner: ({}, {})".format(bx1, by1))
    print("Bottom-right corner: ({}, {})".format(bx2, by2))
    print("lines shape: ", lines.shape)
    plt.figure()
    plt.imshow(im)
    for i in range(lines.shape[1]):
        if lines[0, i] < lines[1, i]:
            pt1 = np.cross(np.array([1, 0, -bx1]), lines[:, i])
            pt2 = np.cross(np.array([1, 0, -bx2]), lines[:, i])
        else:
            pt1 = np.cross(np.array([0, 1, -by1]), lines[:, i])
            pt2 = np.cross(np.array([0, 1, -by2]), lines[:, i])
        pt1 = pt1 / pt1[2]
        pt2 = pt2 / pt2[2]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g')
    # print(vp[0] / vp[2])
    plt.plot(vp[0] / vp[2], vp[1] / vp[2], 'ro')
    plt.show()


# In[367]:


def get_top_and_bottom_coordinates(im, obj):
    """
    For a specific object, prompts user to record the top coordinate and the bottom coordinate in the image.
    Inputs:
        im: np.ndarray of shape (height, width, 3)
        obj: string, object name
    Returns:
        coord: np.ndarray of shape (3, 2)
            where coord[:, 0] is the homogeneous coordinate of the top of the object and coord[:, 1] is the homogeneous
            coordinate of the bottom
    """
    if obj == 'person':
        return np.array([
            [625.0483871 , 622.98387097],
            [465.04516129, 510.46451613],
            [  1.        ,   1.        ]
        ])
    if obj == 'CSL building':
        return np.array([
            [507.37096774, 505.30645161],
            [ 95.49677419, 297.81935484],
            [  1.        ,   1.        ]
        ])
    if obj == 'the spike statue':
        return np.array([
            [600.27419355, 598.20967742],
            [188.4       , 469.17419355],
            [  1.        ,   1.        ]
        ])
    if obj == 'the lamp posts':
        return np.array([
            [290.59677419, 298.85483871],
            [386.59354839, 516.65806452],
            [  1.        ,   1.        ]
        ])
    if obj == 'CSL building South':
        return np.array([
            [897.56451613, 889.30645161],
            [101.69032258, 339.10967742],
            [  1.        ,   1.        ]
        ])
    
    plt.figure()
    plt.imshow(im)

    print('Click on the top coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x1, y1 = clicked[0]
    # Uncomment this line to enable a vertical line to help align the two coordinates
    # plt.plot([x1, x1], [0, im.shape[0]], 'b')
    print('Click on the bottom coordinate of %s' % obj)
    clicked = plt.ginput(1, timeout=0, show_clicks=True)
    x2, y2 = clicked[0]

    plt.plot([x1, x2], [y1, y2], 'b')


    return np.array([[x1, x2], [y1, y2], [1, 1]])


# # Your implementation

# In[368]:


def get_vanishing_point(lines):
    """
    Solves for the vanishing point using the user-input lines.
    """
    # <YOUR IMPLEMENTATION>
    # print("lines: ", lines)
    A = np.array(lines)
    # print("A: ", A)
    # print("np.linalg.svd(A):", np.linalg.svd(A))
    vanishing_point_homogeneous = np.linalg.svd(A)[0][:,-1]
    # print("vanishing_point_homogeneous: ",vanishing_point_homogeneous)
    vanishing_point = vanishing_point_homogeneous / vanishing_point_homogeneous[-1]
    print("vanishing_point: ", vanishing_point)
    return vanishing_point


# In[369]:


def get_horizon_line(vpts):
    """
    Calculates the ground horizon line.
    """
    # <YOUR IMPLEMENTATION>
    # vp1 = vpts[:, 0]
    # print("vp1: ", vp1)
    # vp3 = vpts[:, 2]
    # print("vp3: ", vp3)
    vp1, vp2, vp3 = vpts.T

    # Calculate the slope and intercept of the line connecting vp1 and vp3
    slope = (vp3[1] - vp1[1]) / (vp3[0] - vp1[0])
    intercept = vp1[1] - slope * vp1[0]
    
    # Coefficients of the line equation ax + by + c = 0
    a = slope
    b = -1
    c = intercept
    
    # Normalize the coefficients to ensure the same orientation (a^2 + b^2 = 1)
    norm = np.sqrt(a**2 + b**2)
    a /= norm
    b /= norm
    c /= norm
    print("Horizon Line Parameter: " ,a,b,c)
    return a, b, c
    


# In[370]:


def plot_horizon_line(im, horizon_line, vpts):
    """
    Plots the horizon line.
    """
    # <YOUR IMPLEMENTATION>
    y_vals = np.linspace(vpts[1,0]/vpts[2,0], vpts[1,2]/vpts[2,2], 100)
    x_vals = (-horizon_line[1] * y_vals - horizon_line[2]) / horizon_line[0]
    plt.plot(vpts[0,0]/vpts[2,0], vpts[1,0]/vpts[2,0], 'ro', label='Vanishing points')
    plt.plot(vpts[0,2]/vpts[2,2], vpts[1,2]/vpts[2,2], 'ro')
    plt.imshow(im)
    plt.plot(x_vals, y_vals, color='red', label='Horizon Line')
    plt.legend()
    plt.show()


# In[371]:


def get_camera_parameters(vpts):
    """
    Computes the camera parameters. Hint: The SymPy package is suitable for this.
    """
    # <YOUR IMPLEMENTATION>
    # vp1, vp2, vp3 = vpts.T
    vp1, vp2, vp3 = map(sp.Matrix, vpts.T)
    f, px, py = sp.symbols('f px py', real=True)
    K = sp.Matrix([
        [f, 0, px],
        [0, f, py],
        [0, 0, 1]
    ])
    constraints = [
        sp.Eq(vp1.dot(K.inv().transpose() * K.inv() * vp2), 0),
        sp.Eq(vp1.dot(K.inv().transpose() * K.inv() * vp3), 0),
        sp.Eq(vp2.dot(K.inv().transpose() * K.inv() * vp3), 0)
    ]

    solution = sp.solve(constraints, (f, px, py), dict=True)

    # Extract the solution
    if solution:
        f_value = solution[0][f]
        px_value = solution[0][px]
        py_value = solution[0][py]
        print("f:", f_value)
        print("px:", px_value)
        print("py:", py_value)
        return f_value, px_value, py_value
    else:
        print("No solution found.")
        return None


# In[372]:


def get_rotation_matrix(f ,px, py, vpts):
    """
    Computes the rotation matrix using the camera parameters.
    """
    # <YOUR IMPLEMENTATION>
    K = sp.Matrix([
        [f, 0, px],
        [0, f, py],
        [0, 0, 1]
    ])

    vpts = map(sp.Matrix, vpts.T)

    ri_list = [K.inv() * v for v in vpts]

    # Normalize each column of ri
    normalized_ri_list = [ri / sp.sqrt(sum(x**2 for x in ri)) for ri in ri_list]
    print(normalized_ri_list)
    R = sp.Matrix(normalized_ri_list).transpose()
    print("R", R)
    
    return R


# # Helper Function

# In[373]:


# def project_to_image_plane(point, f, px, py, R):
#     """
#     Projects a 3D point to the image plane using the camera parameters and rotation matrix.

#     Inputs:
#         point: np.ndarray of shape (3, )
#             Homogeneous coordinates of the 3D point.
#         f: float
#             Focal length.
#         px: float
#             Principal point x-coordinate.
#         py: float
#             Principal point y-coordinate.
#         R: np.ndarray of shape (3, 3)
#             Rotation matrix.

#     Returns:
#         projection: np.ndarray of shape (2, )
#             Image plane coordinates.
#     """
#     # Apply rotation
#     rotated_point = R @ point

#     # Perspective projection
#     projection = np.array([rotated_point[0] * f / rotated_point[2] + px,
#                            rotated_point[1] * f / rotated_point[2] + py])

#     return projection

def calculate_intersection(line1, line2):
    """
    Calculate the intersection point of two lines represented by homogeneous coordinates.

    Inputs:
        line1: np.ndarray of shape (3, )
            Parameters of the first line.
        line2: np.ndarray of shape (3, )
            Parameters of the second line.

    Returns:
        intersection_point: np.ndarray of shape (3, )
            Homogeneous coordinates of the intersection point.
    """
    intersection_point = np.cross(line1, line2)
    intersection_point = intersection_point / intersection_point[-1]
    return intersection_point

def line_through_points(point1, point2):
    """
    Calculate the equation of the line passing through two points.

    Inputs:
        point1: np.ndarray of shape (3, )
            Homogeneous coordinates of the first point.
        point2: np.ndarray of shape (3, )
            Homogeneous coordinates of the second point.

    Returns:
        line: np.ndarray of shape (3, )
            Parameters of the line equation.
    """
    line = np.cross(point1, point2)
    return line

def find_farthest_point(reference_bottom_point, reference_top_point):
    """
    Find the point on the line passing through reference_bottom_point and reference_top_point
    that is farthest from reference_bottom_point.

    Inputs:
        reference_bottom_point: np.ndarray of shape (3, )
            Homogeneous coordinates of the reference bottom point.
        reference_top_point: np.ndarray of shape (3, )
            Homogeneous coordinates of the reference top point.

    Returns:
        farthest_point: np.ndarray of shape (3, )
            Homogeneous coordinates of the farthest point.
    """
    # Direction vector of the line
    line_direction = reference_top_point - reference_bottom_point
    scale = np.sqrt(line_direction[0]**2+line_direction[1]**2)
    line_direction = line_direction/scale

    # # Scaling factor to make the point farthest from reference_bottom_point
    # scaling_factor = 1.0  # You can adjust this based on your needs

    # # Calculate the farthest point
    # farthest_point = reference_bottom_point + scaling_factor * line_direction
    print("farthest_point: ", line_direction)
    return line_direction

def plot_line(point1, point2, label=None, color='b'):
    """
    Plot a line passing through two points.

    Inputs:
        point1, point2: np.ndarray of shape (3, )
            Homogeneous coordinates of the two points.
        label: str, optional
            Label for the line in the plot.
        color: str, optional
            Color of the line.

    Returns:
        None
    """
    x_vals = [point1[0]/point1[2], point2[0]/point2[2]]
    y_vals = [point1[1]/point1[2], point2[1]/point2[2]]
    plt.plot(x_vals, y_vals, color, label=label)

def calculate_cross_ratio(point1, point2, point3, point4, transformation_matrix=None):
    """
    Calculates the cross-ratio of four collinear points in projective geometry.

    Inputs:
        point1, point2, point3, point4: np.ndarray of shape (3, )
            Homogeneous coordinates of the four collinear points.
        transformation_matrix: np.ndarray, optional
            Transformation matrix for a projective transformation.

    Returns:
        cross_ratio: float
            The cross-ratio of the four points.
    """
    if transformation_matrix is not None:
        # Apply the projective transformation if provided
        point1 = np.dot(transformation_matrix, point1)
        point2 = np.dot(transformation_matrix, point2)
        point3 = np.dot(transformation_matrix, point3)
        point4 = np.dot(transformation_matrix, point4)
    # Calculate the cross-ratio
    matrix1 = np.vstack((point1, point2, [0, 0, 1]))
    matrix2 = np.vstack((point3, point4, [0, 0, 1]))
    matrix3 = np.vstack((point1, point3, [0, 0, 1]))
    matrix4 = np.vstack((point2, point4, [0, 0, 1]))

    det1 = np.linalg.det(matrix1)
    det2 = np.linalg.det(matrix2)
    det3 = np.linalg.det(matrix3)
    det4 = np.linalg.det(matrix4)

    # Compute the cross-ratio
    cross_ratio = (det1 * det2) / (det3 * det4)

    return cross_ratio


# In[374]:


def estimate_height(im, coords, horizon_line, f, px, py, R, obj, vpts, ref_obj='person', known_reference_height = 1.8):
    """
    Estimates height for a specific object using the recorded coordinates. You might need to plot additional images here for
    your report.
    """
    # <YOUR IMPLEMENTATION>
    # Extract coordinates
    reference_obj_coords = coords[ref_obj]
    object_coords = coords[obj]

    # Extract coordinates
    reference_top_point = reference_obj_coords[:, 0]
    reference_bottom_point = reference_obj_coords[:, 1]
    top_point = object_coords[:, 0]
    bottom_point = object_coords[:, 1]

    # Step 1: Calculate Intersection of Bottom Line with Horizon Line
    intersection_bottom = calculate_intersection(line_through_points(bottom_point,reference_bottom_point), horizon_line)

    # Step 2: Calculate Intersection of Top Line with Reference Object Line
    intersection_top_reference = calculate_intersection(line_through_points(top_point, intersection_bottom), line_through_points(reference_bottom_point, reference_top_point))

    # Step 3: Calculate Cross-Ratio
    cross_ratio = calculate_cross_ratio(reference_bottom_point, intersection_top_reference, reference_top_point, find_farthest_point(reference_bottom_point, reference_top_point))

    # Step 4: Estimate Height
    # Assume known reference object height in the real world (you may adjust this based on your specific scenario)
    # known_reference_height = 12.1  # meters
    estimated_height = known_reference_height * cross_ratio

    # plt.plot(reference_bottom_point[0]/reference_bottom_point[2], reference_top_point, 'ro', label='Vanishing points')
    # plt.plot(vpts[0,2]/vpts[2,2], vpts[1,2]/vpts[2,2], 'ro')
    plt.figure(obj)
    plt.imshow(im)
    # print("reference_bottom_point[0]/reference_bottom_point[2]: ",reference_bottom_point[0]/reference_bottom_point[2])
    plot_line(reference_bottom_point, intersection_top_reference, color='green')
    plot_line(reference_bottom_point, reference_top_point, label='reference', color='red')
    plot_line(bottom_point, top_point, label=obj, color='red')
    
    plot_line(bottom_point, reference_bottom_point)
    plot_line(top_point, intersection_top_reference)
    plot_line(intersection_bottom, reference_bottom_point)
    plot_line(intersection_bottom, intersection_top_reference)
    plot_line(intersection_bottom, reference_top_point, color='pink')

    y_vals = np.linspace(vpts[1,0]/vpts[2,0], vpts[1,2]/vpts[2,2], 100)
    x_vals = (-horizon_line[1] * y_vals - horizon_line[2]) / horizon_line[0]
    plt.plot(vpts[0,0]/vpts[2,0], vpts[1,0]/vpts[2,0], 'ro', label='Vanishing points')
    plt.plot(vpts[0,2]/vpts[2,2], vpts[1,2]/vpts[2,2], 'ro')
    plt.plot(x_vals, y_vals, color='red', label='Horizon Line')
    # plt.plot(intersection_bottom[0]/intersection_bottom[2], intersection_bottom[1]/intersection_bottom[2], )
    plt.legend()
    plt.show()

    return estimated_height


# # Main function

# 

# 

# In[375]:


im = np.asarray(Image.open('CSL.jpeg'))

# Part 1
# Get vanishing points for each of the directions
num_vpts = 3
vpts = np.zeros((3, num_vpts))
for i in range(num_vpts):
    print('Getting vanishing point %d' % i)
    # Get at least three lines from user input
    n, lines, centers = get_input_lines(im, i)
    # <YOUR IMPLEMENTATION> Solve for vanishing point
    vpts[:, i] = get_vanishing_point(lines)
    # Plot the lines and the vanishing point
    # plot_lines_and_vp(im, lines, vpts[:, i])

# <YOUR IMPLEMENTATION> Get the ground horizon line
horizon_line = get_horizon_line(vpts)
# <YOUR IMPLEMENTATION> Plot the ground horizon line
# plot_horizon_line(im, horizon_line, vpts)




# In[376]:


# Part 2
# <YOUR IMPLEMENTATION> Solve for the camera parameters (f, u, v)
f ,px, py = get_camera_parameters(vpts)


# Part 3
# <YOUR IMPLEMENTATION> Solve for the rotation matrix
R = get_rotation_matrix(f ,px, py, vpts)


# In[377]:


# Part 4
# Record image coordinates for each object and store in map
objects = ('person', 'CSL building', 'the spike statue', 'the lamp posts', 'CSL building South','window')
# objects = ('person', 'CSL building', 'the spike statue', 'the lamp posts')

coords = dict()
for obj in objects:
    coords[obj] = get_top_and_bottom_coordinates(im, obj)
print("coords: ", coords)
# <YOUR IMPLEMENTATION> Estimate heights

# for obj in objects[1:]:
for obj in ('person', 'CSL building', 'the lamp posts', 'CSL building South','window'):

    print('Estimating height of %s' % obj)
    height = estimate_height(im, coords, horizon_line, f, px, py, R, obj, vpts, ref_obj='the spike statue', known_reference_height = 12)
    print("Height of ",obj," is ", height)


# In[ ]:




