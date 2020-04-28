# Import required packages:
import cv2
import numpy as np


class FaceProcessing(object):
   def __init__(self):
        self.file1 = "/home/somashekhar13/mysite/deploy.prototxt"
        self.file2 = "/home/somashekhar13/mysite/res10_300x300_ssd_iter_140000.caffemodel"
        self.file3 = "/home/somashekhar13/mysite/age_deploy.prototxt"
        self.file4 = "/home/somashekhar13/mysite/age_net.caffemodel"
        self.prototxtPath = self.file1
        self.weightsPath = self.file2
        self.ageprototxtPath =self.file3
        self.ageweightsPath = self.file4

   def face_detection(self, imaged):
        output=[]
        AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(38-43)", "(48-53)", "(60-100)"]
        faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)
        ageNet = cv2.dnn.readNet(self.ageprototxtPath, self.ageweightsPath)
        image_array = np.asarray(bytearray(imaged), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        for i in range(0, detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence > 0.5:
              box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
              (startX, startY, endX, endY) = box.astype("int")
              face = image[startY:endY, startX:endX]
              faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)
              ageNet.setInput(faceBlob)
              preds = ageNet.forward()
              i = preds[0].argmax()
              age = AGE_BUCKETS[i]
              ageConfidence = preds[0][i]
              y = startY - 10 if startY - 10 > 10 else startY + 10
              face = {"box": [int(startX), int(startY),int(endX),int(endY),int(y),age,str(ageConfidence)]}
              output.append(face)
        return output




