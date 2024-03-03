import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
from skimage.transform import resize
import time


def origin(image):
    print("***********************Doing original image***********************")
    return image

def shift_left(image, shift_percent=0.2):
    print("**********************Doing shift left image**********************")
    height, width = image.shape[:2]
    shift_pixels = int(width * shift_percent)
    shifted_image = image[:, shift_pixels:]
    return shifted_image

def shift_right(image, shift_percent=0.2):
    print("**********************Doing shift right image**********************")
    height, width = image.shape[:2]
    shift_pixels = int(width * shift_percent)
    shifted_image = image[:, :width - shift_pixels]
    return shifted_image

def rotate_counterclockwise(image):
    print("**************Doing counter clockwise rotation image**************")
    rotated_image = cv2.transpose(image)
    rotated_image = cv2.flip(rotated_image, 0)
    return rotated_image

def rotate_clockwise(image):
    print("******************Doing clockwise rotation image******************")
    rotated_image = cv2.transpose(image)
    rotated_image = cv2.flip(rotated_image, 1)
    return rotated_image

def enlarge_and_crop(image, scale_factor=2):
    print("*******************Doing enlarge and crop image*******************")
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    
    crop_x = (new_width - width) // 2
    crop_y = (new_height - height) // 2
    cropped_image = resized_image[crop_y:crop_y+height, crop_x:crop_x+width]
    
    return cropped_image

def imshowImage(image, scale_percent=300):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)

    resized_image = cv2.resize(image, (width, height))
    
    cv2.imshow('image',resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def non_max_suppression(corners, threshold):
    # Apply non-maximum suppression
    local_max = cv2.dilate(corners, None)
    corner_mask = (corners == local_max)
    corners[corner_mask & (corners >= threshold * corners.max())] = 255
    corners[~corner_mask] = 0
    return corners

def compute_orientation_histogram(window, num_bins=36):
    # Compute gradient magnitude and direction in the window
    # print("windows size: ", window.shape)
    gradient_y, gradient_x = np.gradient(window)
    # print(gradient_y)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = np.arctan2(gradient_y, gradient_x)
    orientation = np.where(abs(orientation - np.pi) < 1e-10, -np.pi, orientation)

    # Create a histogram of orientations weighted by magnitude
    hist, bin_edges = np.histogram(orientation, bins=num_bins, range=(-np.pi, np.pi),weights=magnitude)

    # Find the most common orientation bin
    most_common_orientation_bin = bin_edges[np.argmax(hist)]
    most_common_orientation_bin = -most_common_orientation_bin
    return most_common_orientation_bin

def compute_most_common_orientation(image, x, y, window_size):
    # Ensure the window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Extract the window around the corner
    corner_window = image[y - window_size // 2:y + window_size // 2 + 1, x - window_size // 2:x + window_size // 2 + 1]
    if corner_window.shape[0] != window_size or corner_window.shape[1] != window_size:
        print("window outbound image!")
        return None
    most_common_orientation = compute_orientation_histogram(corner_window)

    return most_common_orientation

# mark the corner and its property on the original colored image
def mark_corner(image_colored,best_scale,x,y,orientation_angle,circle_scale,draw_scale,mark_width):
    cv2.putText(image_colored, f'{best_scale:d}/Scale,({x:d},{y:d})/Loc', (x + circle_scale*2, y + circle_scale*2), cv2.FONT_HERSHEY_SIMPLEX, circle_scale*0.2 , (255, 0, 255), mark_width)
    cv2.circle(image_colored, (x, y), best_scale*circle_scale, (0, 255, 0), mark_width)
    cv2.line(image_colored, (x, y), (int(x + draw_scale*best_scale * np.cos(orientation_angle)), int(y - draw_scale*best_scale * np.sin(orientation_angle))), (200, 100, 0), mark_width)
    
    return image_colored


def corner_image_process(file_name, threshold):
    image = cv2.imread('data/'+file_name+'.jpg', cv2.IMREAD_GRAYSCALE)
    image_colored = cv2.imread('data/'+file_name+'.jpg')
    min_side_width = min(image.shape[1],image.shape[0])
    min_side_width = max(min_side_width, 1300)
    circle_scale = int(min_side_width/300)
    draw_scale = int(min_side_width/40)
    mark_width = int(min_side_width/450)
    transformed_functions = [origin,shift_left, shift_right, rotate_counterclockwise, rotate_clockwise, enlarge_and_crop]
    transformed_functions = [shift_left, shift_right, rotate_counterclockwise, rotate_clockwise, origin, enlarge_and_crop]

    # transformed_functions = [origin, rotate_counterclockwise]
    for transformation in transformed_functions:
        transformed_image_gray = transformation(image.copy())
        transformed_image_colored = transformation(image_colored.copy())

        # Compute Harris corners
        corners = cv2.cornerHarris(transformed_image_gray, blockSize=3, ksize=3, k=0.1)

        corners = non_max_suppression(corners, threshold=threshold) # Adjust this threshold as needed

        # Define scales for LOG filters
        scales = [ 1,  2, 3, 4, 5, 6, 7]  # You can adjust these scales
        corners_count = 0
        # print(corners.shape)
        for y in range(corners.shape[0]):
            for x in range(corners.shape[1]):
                if corners[y, x] == 255:
                    corners_count += 1
                    print("corner location: ", x, y)
                    max_response = 0
                    best_scale = 1

                    transformed_image_gray_rescale = transformed_image_gray/255.0

                    # Compute LOG responses at different scales
                    for scale in scales:
                        # scaled_image = cv2.GaussianBlur(transformed_image_gray, (0, 0), scale)
                        # scaled_image = transformed_image_gray - scaled_image
                        # log_response = gaussian_laplace(scaled_image, scale)

                        
                        log_response = gaussian_laplace(transformed_image_gray_rescale, scale)
                        log_response = log_response * (scale ** 2)

                        
                        response = abs(log_response[y, x])
                        # print(response)

                        if response > max_response:
                            max_response = response
                            best_scale = scale

                    # Compute orientation
                    # Create a window around the corner at the selected scale
                    window_size = int(6 * best_scale)
                    orientation_angle = compute_most_common_orientation(transformed_image_gray_rescale, x, y, window_size)
                    if orientation_angle == None:
                        continue
                    # Mark the corner, scale, and orientation on the image
                    # print("orientation_angle: ",orientation_angle)
                    transformed_image_colored = mark_corner(transformed_image_colored,best_scale,x,y,orientation_angle,circle_scale,draw_scale,mark_width)
        print("corner count: ", corners_count)



        # for x, y, radius in blob_coordinates:
        #     # cv2.circle(image_colored, (y, x), radius, (0, 0, 255), 2)
        #     # cv2.arrowedLine(image_colored, (y, x), (y + radius, x), (0, 255, 0), 2)
        #     mark_corner(image_colored,radius,x,y,orientation_angle,circle_scale,draw_scale,mark_width)
        # print("corner count: ", corners_count)
        # mark_corner(image_colored,best_scale,x,y,orientation_angle,circle_scale,draw_scale,mark_width)
        # imshowImage(transformed_image_colored,show_scale)
        cv2.imwrite('corner_'+file_name+'_'+transformation.__name__+'.jpg', transformed_image_colored)
        print("Image saved successfully.")


def main():
    
    # file_name = "blackandwhite"
    # file_name = "blackandwhite1"
    # file_name = "chess"
    # file_name = "desk"
    # file_name = "stapler"
    # # file_name = "icon1"
    # file_name = "icon2"
    # # file_name = "logo"
    # file_name = "logo2"
    # file_name = "merged_00149v"
    # file_name = "merged_01047u"
    # file_name = "merged_01861a"
    file_names = ["icon1","icon2","desk","merged_00149v","merged_01047u","merged_01861a"]
    file_names = ['brick','floor','siebel','speedlimit']
    file_names = ['floor']
    # file_names = ['merged_00149v','merged_01047u','merged_01861a']
    # file_names = ['merged_00149v']
    file_names = ['blackandwhite1']
    # file_names = ['dog']
    # file_names = ['switch']
    # file_names = ['light']

    threshold = 1
    show_scale = 30
    for file_name in file_names:

        print('processing file: ', file_name)
        start_time = time.time()
        corner_image_process(file_name, threshold)
        end_time = time.time()
        total_time = end_time - start_time 
        print('[Info]Total time: ',total_time)




    

if __name__ == "__main__":
    main()
