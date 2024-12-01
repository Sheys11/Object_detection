import cv2
import imutils

# Load the pre-trained deep neural network (DNN) model
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.cfg", "dnn_model/yolov4-tiny.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(500, 500), scale=1/255)

# Load class names for object detection
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

    while True:
        _, frame = cam.read()
        frame = imutils.resize(frame, height=450 , width=900)

        # Perform object detection
        (class_ids, scores, bboxes) = model.detect(frame)

        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
        class_name = classes[class_id]
    
        # Annotate the frame
        cv2.putText(frame, class_name, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
    
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 4)
    
        cv2.imshow('object detection', frame)
        cv2.waitKey(1)
