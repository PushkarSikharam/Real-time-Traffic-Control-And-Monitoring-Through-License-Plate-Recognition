import cv2
import imutils
import numpy as np
import pytesseract

# Set the tesseract_cmd path if Tesseract is not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image (ensure the file exists in the same directory)
image_path = '1.jpg'  # Replace with your image file path
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"The file '{image_path}' was not found. Please check the path.")

# Resize the image for uniform processing
img = cv2.resize(img, (600, 400))

# Convert to grayscale and apply bilateral filter
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)

# Detect edges using Canny edge detector
edged = cv2.Canny(gray, 30, 200)

# Find contours
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# Loop through contours to find the license plate area
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:  # Look for a rectangle
        screenCnt = approx
        break

# Check if a license plate-like contour was detected
if screenCnt is None:
    print("No contour detected")
else:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    # Mask the region of interest and extract the license plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Crop the license plate from the image
    (x, y) = np.where(mask == 255)
    if x.size > 0 and y.size > 0:  # Ensure cropping is possible
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        # Perform OCR on the cropped license plate
        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        print("Detected license plate Number is:", text)

        # Display the images
        img = cv2.resize(img, (500, 300))
        Cropped = cv2.resize(Cropped, (400, 200))
        gray = cv2.resize(gray, (500, 300))
        edged = cv2.resize(edged, (500, 300))

        cv2.imshow('Car Image', img)
        cv2.imshow('Cropped License Plate', Cropped)
        cv2.imshow('Gray Image', gray)
        cv2.imshow('Edge Detection', edged)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not isolate the license plate area.")
