import cv2
img = cv2.imread('assets/scottylabs.png', 0)

cv2.imshow('hi', img)
cv2.waitKey(0)
cv2.destroyAllWindows