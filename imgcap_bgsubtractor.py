import numpy as np
import cv2

cv2.__version__

# 選擇第二隻攝影機
cap = cv2.VideoCapture(0)

fgbg =  cv2.bgsegm.createBackgroundSubtractorMOG()

while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()

  fgmask = fgbg.apply(frame)
    
  # 顯示圖片
  cv2.imshow('frame', fgmask)

  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
