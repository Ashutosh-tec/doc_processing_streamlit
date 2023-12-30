###################################
# This code detects face using cv2's Cascade function.
###################################

import cv2

def face_detect_cv2(img)->None:
    """
    Detect face from image
    img: opened by cv2
    """
    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert the image to grayscale (required for face detection)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    load_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(load_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # cv2.imwrite("output.jpeg", img=img)
    return load_img

# img_path = 'D:/Codes/Experiments/cv_project_colab_december/final_project/task_processor/face_detection_cv2/sample_image/images.jpeg'
# face_detect_cv2(img = cv2.imread(img_path))
