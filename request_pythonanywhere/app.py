"""
 Request example to perform a POST request in order to detect and draw the faces in a image using the
 API hosted at pythonanywhere
"""

#USAGE
# python app.py

# Import required packages:
import cv2
import requests

FACE_DETECTION_REST_API_URL = "http://somashekhar13.pythonanywhere.com/detect"
IMAGE_PATH = "bikku.jpeg"

# Load the image and construct the payload:
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# Submit the POST request:
r = requests.post(FACE_DETECTION_REST_API_URL, files=payload)
print(r)

# See the response:
print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))

# Get JSON data from the response and get 'result':
json_data = r.json()
result = json_data['result']
print(result)

# Opening the image in opencv mode to draw the bounding boxes
img=cv2.imread("bikku.jpeg")

# Draw faces in the OpenCV image:
for face in result:
    startX,startY,endX,endY,y,age,ageConfidence = face['box']
    text = "{}: {:.2f}%".format(age, float(ageConfidence) * 100)
    print(text)
    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Image", img)
cv2.waitKey(0)
