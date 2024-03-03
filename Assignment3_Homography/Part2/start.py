# imports
import os
import sys
import glob
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import time

#####################################
### Provided functions start here ###
#####################################

# Image loading and saving

def LoadFaceImages(pathname, subject_name, num_images):
    """
    Load the set of face images.  
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs

def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,0]*128+128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,1]*128+128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,2]*128+128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)


# Plot the height map

def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')

    surf = ax.plot_surface(H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)
    plt.show()


# Plot the surface normals

def plot_surface_normals(surface_normals):
    """
    surface_normals: h x w x 3 matrix.
    """
    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:,:,0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:,:,1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:,:,2])


#######################################
### Your implementation starts here ###
#######################################

def preprocess(ambimage, imarray):
    """
    preprocess the data: 
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """
    # Subtract ambient image from each image
    processed_imarray = imarray - ambimage[:, :, np.newaxis]
    
    # Make sure no pixel is less than zero
    processed_imarray[processed_imarray < 0] = 0
    
    # Rescale values to be between 0 and 1
    processed_imarray = processed_imarray / 255.0
    print("processed_imarray: ", processed_imarray.shape)
    return processed_imarray


def photometric_stereo(imarray, light_dirs):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """
    # Flatten the image array to a 2D array
    h, w, Nimages = imarray.shape
    imarray = imarray.reshape((h * w, Nimages))
    
    # Solve for albedo using the least-squares method
    albedo_image = np.linalg.norm(imarray, axis=1).reshape((h, w))
    albedo_image /= np.max(albedo_image)
    # Normalize the light directions
    light_dirs /= np.linalg.norm(light_dirs, axis=1)[:, np.newaxis]
    
    # Solve for surface normals
    surface_normals = np.linalg.lstsq(light_dirs, imarray.T, rcond=None)[0]
    surface_normals = surface_normals.T.reshape((h, w, 3))
    # Display the image or 2D array
    plt.imshow(albedo_image, cmap='gray')  # 'gray' cmap is for grayscale images, use 'jet' for colored heatmaps, for example

    # Add a title to the plot (optional)
    plt.title("My Image")

    # Show the plot (this will open a window with the image)
    plt.show()
    print("albedo_image:", albedo_image.shape)
    print("surface normals:", surface_normals.shape)
    return albedo_image, surface_normals

def integrate_rows(surface_normals):
    fx = surface_normals[:, :, 0] / surface_normals[:,:,2]
    fy = surface_normals[:, :, 1] / surface_normals[:,:,2]
    row_sum = np.cumsum(fx, axis=1)
    col_sum = np.cumsum(fy, axis=0)
    height_map = col_sum + row_sum[0, :][np.newaxis, :]


    return height_map

def integrate_columns(surface_normals):
    fx = surface_normals[:, :, 0] / surface_normals[:,:,2]
    fy = surface_normals[:, :, 1] / surface_normals[:,:,2]
    row_sum = np.cumsum(fx, axis=1)
    col_sum = np.cumsum(fy, axis=0)
    height_map = row_sum + col_sum[:, 0][:, np.newaxis]
    return height_map


# def get_random_neighbour(i, j, h, w):
#     while True:
#         direction = np.random.choice(['up', 'down', 'left', 'right'])
#         if direction == 'up':
#             if i > 0:
#                 return i - 1, j
#         elif direction == 'down':
#             if i < h - 1:
#                 return i + 1, j
#         elif direction == 'left':
#             if j > 0:
#                 return i, j - 1
#         elif direction == 'right':
#             if j < w - 1:
#                 return i, j + 1

def integrate_random_paths(surface_normals, num_paths=10):
    h, w, _ = surface_normals.shape
    height_map = np.zeros((h, w))
    fx = surface_normals[:, :, 0] / surface_normals[:,:,2]
    fy = surface_normals[:, :, 1] / surface_normals[:,:,2]

    for y in range(surface_normals.shape[0]):
        for x in range(surface_normals.shape[1]):
            for path in range(num_paths):
                path_length = y + x
                path_indices = np.random.choice([0, 1], path_length)
                # print(path_indices)
                sum = 0
                current_x = 0 
                current_y = 0

                for direction in path_indices:
                    # print("direction", direction)
                    # print("current x", current_x)
                    # print("x",x)
                    if (direction == 0 or (direction == 1 and current_y >= y)) and current_x < x:
                        sum += fx[current_y, current_x]
                        current_x += 1
                    elif (direction == 1 or (direction == 0 and current_x >= x)) and current_y < y:
                        sum += fy[current_y, current_x]
                        current_y += 1
                    elif current_x < x:
                        print("Error!!")

                height_map[y, x] += sum

    height_map /= num_paths

    return height_map




def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    start_time = time.time()
    h, w, _ = surface_normals.shape
    height_map = np.zeros((h, w))
    
    if integration_method == 'average':
        # Simple average of row and column integration
        height_map = 0.5 * (integrate_rows(surface_normals) + integrate_columns(surface_normals))
    elif integration_method == 'column':
        height_map = integrate_columns(surface_normals)
    elif integration_method == 'row':
        height_map = integrate_rows(surface_normals)
    elif integration_method == 'random':
        # Implement random path integration (you need to define this function)
        height_map = integrate_random_paths(surface_normals)
        print("random")
    print("height map:", height_map.shape)
    end_time = time.time()
    execution_time = (end_time - start_time) * 1e9
    print(f"execution time: {execution_time} ns")
    # Display the image or 2D array
    plt.imshow(height_map, cmap='gray')  # 'gray' cmap is for grayscale images, use 'jet' for colored heatmaps, for example

    # Add a title to the plot (optional)
    plt.title("Height Map")

    # Show the plot (this will open a window with the image)
    plt.show()
    return height_map



# Main function
if __name__ == '__main__':
    root_path = 'croppedyale/'
    subject_name = 'yaleB02'
    integration_method = 'random'
    save_flag = True

    full_path = '%s%s' % (root_path, subject_name)
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path, subject_name,
                                                        64)

    processed_imarray = preprocess(ambient_image, imarray)

    albedo_image, surface_normals = photometric_stereo(processed_imarray,
                                                    light_dirs)

    height_map = get_surface(surface_normals, integration_method)

    if save_flag:
        save_outputs(subject_name, albedo_image, surface_normals)

    plot_surface_normals(surface_normals)

    display_output(albedo_image, height_map)





