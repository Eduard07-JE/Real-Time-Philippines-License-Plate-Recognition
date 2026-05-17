import cv2
from ultralytics import YOLO
import easyocr

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Load models
vehicle_model = YOLO("./models/yolov8n.pt")
plate_model = YOLO("./models/best.pt")

# Load image
img = cv2.imread("lto_plate_number.jpg")

if img is None:
    print("Image not found!")
    exit()

# Vehicle detection
vehicle_results = vehicle_model(img)[0]

for box in vehicle_results.boxes.data.tolist():
    x1, y1, x2, y2, conf, cls = box

    vehicle_crop = img[int(y1):int(y2), int(x1):int(x2)]

    # Plate detection inside vehicle
    plate_results = plate_model(vehicle_crop)[0]

    print("PLATE DETECTIONS:", len(plate_results.boxes.data.tolist()))

    for pbox in plate_results.boxes.data.tolist():
        px1, py1, px2, py2, pconf, pcls = pbox

        plate_crop = vehicle_crop[int(py1):int(py2), int(px1):int(px2)]

        # Convert to grayscale (important for OCR)
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

        # EasyOCR
        result = reader.readtext(gray, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

        if result:
            text = result[0][1].upper()
            print("DETECTED PLATE:", text)

        # Show crop for debugging
        cv2.imshow("Plate", plate_crop)
        cv2.waitKey(0)

cv2.destroyAllWindows()