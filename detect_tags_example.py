import cv2

# LOAD IMAGE AND TAG DICTIONARY
tags = cv2.imread('data/two_tags_ARUCO_ORIGINAL.png')
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)

# DETECT TAGS IN IMAGE
corners, ids, rejects = cv2.aruco.detectMarkers(cv2.cvtColor(tags, cv2.COLOR_BGR2GRAY), arucoDict)

# DRAW DETECTION AND SAVE FILE
detection = cv2.aruco.drawDetectedMarkers(tags, corners, borderColor=(255, 0, 0))
cv2.imwrite('detection_two_tags_ARUCO_ORIGINAL.png', detection)
