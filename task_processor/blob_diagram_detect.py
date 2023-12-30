import cv2
import numpy as np

def create_line_kernel(angle_degrees:int, kernel_length:int, kernel_width:int):
    """
    create a line kernel
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

def diagram_detector(image: np.ndarray, debug: bool=False, output_folder: str=None):
    """
    returns an image, where the diagram has been detected.
    """
    original_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # make it a gray image
    if image.shape[-1] == 2:
        # If 2 channels, assume it's already grayscale
        gray = image
    else:
        # Convert to grayscale if the image has 3 channels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # thresh = cv2.threshold(gray, 0, 255,
    # cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = ~thresh
    img_area = image.shape[0] * image.shape[1]

    img_bin_final = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    img_bin_final_gap_filled = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for angle_degrees in [0, 90, 45, -45]:
        rotated_kernel = create_line_kernel(angle_degrees, int(0.000001 * img_area), int(0.000001 * img_area))
        
        # Apply morphological opening and closing with the rotated kernel
        img_bin_rotated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, rotated_kernel)
        
        # closing_kernel_rotated = np.ones((int(0.000003 * img_area), int(0.000003 * img_area)), np.uint8)
        closing_kernel_rotated = create_line_kernel(angle_degrees, int(0.000001 * img_area), int(0.000001 * img_area))
        closing_img_rotated = cv2.morphologyEx(img_bin_rotated, cv2.MORPH_CLOSE, closing_kernel_rotated)
        
        
        # Merging Horizontal and Vertical Images before gap filling
        img_bin_final = img_bin_final | img_bin_rotated 

        # Merge after gap filling
        img_bin_final_gap_filled = img_bin_final_gap_filled | closing_img_rotated
    blackhat = img_bin_final_gap_filled

    cv2.imwrite(f'{output_folder}/blackhat.png', blackhat) if debug else None

    # remove words 
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(blackhat)
    for label in range(1, num_labels):
        # area of the connected component
        area = stats[label, cv2.CC_STAT_AREA]
        
        width, height = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]

        if area <= 0.95 * width*height \
            and  width * height < 0.000098*img_area:
                x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], width, height
                # cv2.rectangle(img_test, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(blackhat, (x, y), (x + w, y + h), (0, 0, 0), -1) 

                # considering the case where wirds are itself part of the diagram 
                # Calculate the coordinates for the middle line
                middle_x = (x + w) // 2
                middle_y1 = y
                middle_y2 = (y + h)
                # Draw a straight black line in the middle of the black box
                cv2.line(blackhat, (middle_x, middle_y1), (middle_x, middle_y2), (255, 255, 255), 1)  # Black color (0, 0, 0)
    
    cv2.imwrite(f'{output_folder}/detect_words.png', blackhat) if debug else None

    # Apply Sobel edge detection along the x-axis (horizontal gradient)
    grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)

    # Apply Sobel edge detection along the y-axis (vertical gradient)
    grad_y = cv2.Sobel(blackhat, cv2.CV_32F, 0, 1, ksize=-1)

    # Combine the x and y gradients to obtain the magnitude gradient
    grad = cv2.magnitude(grad_x, grad_y)

    # Define the Sobel kernels for diagonal gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # Apply Sobel edge detection along the main diagonal (from top-left to bottom-right)
    grad_diag_main = cv2.filter2D(grad, -1, sobel_x) + cv2.filter2D(grad, -1, sobel_y)

    # Apply Sobel edge detection along the anti-diagonal (from top-right to bottom-left)
    grad_diag_anti = cv2.filter2D(grad, -1, np.flip(sobel_x, axis=1)) + cv2.filter2D(grad, -1, sobel_y)

    # Combine the diagonal gradients to obtain the magnitude gradient
    grad = cv2.magnitude(grad_diag_main, grad_diag_anti)

    cv2.imwrite(f'{output_folder}/sobel.png', grad) if debug else None
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    if maxVal != minVal:
        grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")

    dilate = cv2.dilate(grad, None, iterations=2)
    cv2.imwrite(f'{output_folder}/dilate.png', dilate) if debug else None

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilate)

    for label in range(1, num_labels):
        # area of the connected component
        area = stats[label, cv2.CC_STAT_AREA]
        
        width, height = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        # pass the diagram with area more than 0.00045 times of image area NOTE: 0.00045 was accurate for most CAD's
        if width * height > 0.00155*img_area:
            
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], width, height

        # Draw a rectangle around the connected component
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return original_image


# image = cv2.imread('cad_images/cad2.png')
# detected_img = diagram_detector(image, debug=True, output_folder="temp")
# cv2.imwrite("temp/diagram.png", detected_img)