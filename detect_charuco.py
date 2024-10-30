
import cv2
from cv2 import aruco
import numpy as np

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# 2cm aruco marker in 4cm checkerboard
board = aruco.CharucoBoard((5, 7), 0.04, 0.02, dictionary)

# cm -> inch -> pixel in 300dpi
height = int(7 * 4 / 2.54 * 300)
width = int(5 * 4 / 2.54 * 300)

cv2.imwrite("charuco.jpg", board.generateImage((width, height)))



dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
board = aruco.CharucoBoard((5,7), 0.04, 0.02, dictionary)

params = aruco.DetectorParameters()
detector = aruco.CharucoDetector(board, detectorParams=params)

img_path = '/workspace/data/kandao/KD_20240731_193042_MP4/convert_center_cam/dataset_for_4k4d/images/cam02/000000.jpg'
# img_path = 'gboriginal.jpg'
# img_path = 'charuco.jpg'
img = cv2.imread(img_path)
width = img.shape[1]
height = img.shape[0]

img = cv2.resize(img, None, fx=2, fy=2)
kernel = np.ones((3,3),np.uint8)
img = cv2.erode(img,kernel,iterations = 1)
img = cv2.resize(img, (width, height))

# img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
charuco_corners, charuco_ids, mk_corners, mk_ids = detector.detectBoard(img)
print(charuco_corners.shape[0])
# img_points = ch_corners.reshape(-1, 2)

show = img.copy()
    # (0,255,0)はコーナーに描画する色を指す
show = aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids, (0, 255, 0))
cv2.imwrite('img.png', show)