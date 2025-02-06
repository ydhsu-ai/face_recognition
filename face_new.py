from openvino.runtime import Core

ie = Core()
devices = ie.available_devices
print("Available devices:", devices)

#############################################################

model_path = "face-detection-adas-0001.xml"
compiled_model = ie.compile_model(model_path, "CPU")

# 取得輸入輸出層
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

#############################################################
import cv2
import numpy as np

# 開啟攝影機
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換成 OpenVINO 可處理的格式
    input_blob = np.expand_dims(cv2.resize(frame, (672, 384)), axis=0).transpose((0, 3, 1, 2))

    # 進行人臉偵測
    results = compiled_model([input_blob])[output_layer]

    # 繪製偵測框
    for detection in results[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            xmin, ymin, xmax, ymax = (detection[3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype("int")
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
