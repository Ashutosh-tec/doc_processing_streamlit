"""
#Open the image using cv2
image = cv2.imread(< IMAGE PATH >)
# Run detection algorithm with debug
main(image, < FOLDER >, debug = True)
# Run detection algorithm without debug
main(image)
"""

import os
import cv2
import json
import numpy as np


class Component:
    def __init__(self):
        self.components = dict()

    def add_component(self, label, type, x, y, w, h):
        component_data = {
            'type': type,
            'annotation': [int(x), int(y), int(x + w), int(y + h)] 
        }
        self.components[label] = component_data

    def generate_json(self):
        return json.dumps(self.components, indent=4)

def create_line_kernel(angle_degrees: int, kernel_length:int, kernel_width:int)-> np.ndarray:
    """
    Line Kernel for accuracy in filling.

    Args:
        angle_degrees (int): angle of rotation
        kernel_length (int): length of kernel
        kernel_width (int): width of kernel

    Returns:
        ndarray from numpy, in which rotated line has been drawn.
    """
    # Create a blank image as a kernel
    kernel = np.zeros((kernel_length, kernel_width), dtype=np.uint8)
    # Calculate the center of the kernel
    center_x, center_y = (kernel_width - 1) / 2, (kernel_length - 1) / 2 
    # Calculate the line parameters
    angle_radians = np.radians(angle_degrees)
    line_length = max(kernel_length, kernel_width)  # Ensure line is long enough to cover the entire kernel
    # Calculate the coordinates of the line endpoints
    x1 = int(center_x - 0.5 * line_length * np.cos(angle_radians))
    y1 = int(center_y - 0.5 * line_length * np.sin(angle_radians))
    x2 = int(center_x + 0.5 * line_length * np.cos(angle_radians))
    y2 = int(center_y + 0.5 * line_length * np.sin(angle_radians))  
    # Draw the line on the kernel
    if abs(angle_degrees) == 90:
        cv2.line(kernel, (x1, y1), (x1, y2), 255, thickness=1)
    else:
        cv2.line(kernel, (x1, y1), (x2, y2), 255, thickness=1)
    return kernel


def main(image, folder_name:str= None, debug = False):
    """
    Open image in cv2 format. Perform diagram detection on that.
    NOTE: folder_name is not necessary if debug = False

    Args:
        image : cv2 opened format image.
        folder_name (str): folder you want to save the output. 
        debug (bool): Make it True if you want to see the results stepwise.
    """
    if debug:
        cv2.imwrite(folder_name+"/original_image.png", image)
        cropped_folder = folder_name + "/cropped"
        if not os.path.exists(cropped_folder):
            os.mkdir(cropped_folder)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Threshold the image to create a binary image and inversing to workwith morphology
    thresh = ~cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    img_area = image.shape[0] * image.shape[1]
    # Empty images 
    img_bin_final = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    img_bin_final_gap_filled = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Iterate through a range of angles (from -90 degrees to +90 degrees)
    for angle_degrees in range(-90, 91):
        # Apply morphological opening and closing with the rotated kernel
        rotated_kernel = create_line_kernel(angle_degrees, int(0.00000005*img_area), int(0.00000005*img_area))
        img_bin_rotated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rotated_kernel)
        
        closing_kernel_rotated = create_line_kernel(angle_degrees, int(0.000001 * img_area), int(0.000001 * img_area))
        closing_img_rotated = cv2.morphologyEx(img_bin_rotated, cv2.MORPH_CLOSE, closing_kernel_rotated)
        
        # Merging Horizontal and Vertical Images before gap filling
        img_bin_final = img_bin_final | img_bin_rotated if debug else None

        # Merge after gap filling
        img_bin_final_gap_filled = img_bin_final_gap_filled | closing_img_rotated
    # Perform connected components labeling with stats
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(img_bin_final_gap_filled, connectivity=8)
    # Define a color for marking boundaries (e.g., green)
    boundary_color = (0, 0, 255)

    final_result = Component()
    # Iterate through the connected components (excluding background label)
    for label in range(1, num_labels):
        # Extract the region of interest (ROI) for the connected component
        x, y, w, h, _ = stats[label]       
        # do not include small connected components.
        if (w*h) < 0.00155*img_area:
            continue
        if debug:          
            try:
                # Draw a boundary around the ROI
                cv2.rectangle(image, (x, y), (x + w, y + h), boundary_color, 2)
                cropped_image = image[y:y+h,x:x+w] # Slicing to crop the image
                # Display the cropped image
                cv2.imwrite(cropped_folder+f"/cropped__{label}.png", cropped_image)
            except Exception as e:
                print(e)
        final_result.add_component(label=f'Component_{label}', type = 'Unknown', x=x, y=y, w=w, h=h)

    if debug:
        cv2.imwrite(f'{folder_name}/thresh.png', thresh)
        cv2.imwrite(f'{folder_name}/morph_open.png', img_bin_final)
        cv2.imwrite(f'{folder_name}/morph_close.png', img_bin_final_gap_filled)
        cv2.imwrite(f'{folder_name}/detected_image.png', image)
    return final_result.generate_json(), image

# image = cv2.imread("/mnt/d/Codes/Experiments/cv_project_colab_december/imported_from_itech/dynamic-dgd/sample/original_image.png")
# image = cv2.imread("/mnt/d/Codes/Experiments/cv_project_colab_december/final_project/cad_images/cad2.png")
# print(main(image, "temp", True))
# print(main(image))

