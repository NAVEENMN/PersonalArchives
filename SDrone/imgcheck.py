import cv2
import numpy as np
from matplotlib import pyplot as plt
path = "D:\\map_data\\1496105141199\\1496105155524.png"
oimg = cv2.imread(path,0)
'''
hist = cv2.calcHist([oimg],[0],None,[256],[0,256])
plt.hist(oimg.ravel(),256,[0,256])
plt.title('Histogram for gray scale picture')
plt.show()

while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: break             # ESC key to exit 
cv2.destroyAllWindows()
'''
blur = cv2.GaussianBlur(oimg,(5,5),0)
ret3,l1 = cv2.threshold(blur,52,150,cv2.THRESH_BINARY)
ret3,l2 = cv2.threshold(blur,68,255,cv2.THRESH_BINARY)

res = cv2.addWeighted(l1,0.5,l2,0.5,0)
'''
kernel = np.ones((3,3),np.uint8)
res = cv2.dilate(res,kernel,iterations = 1)
'''
edges = cv2.Canny(oimg,40,80,apertureSize = 3)
res = np.hstack((oimg,res)) #stacking images side-by-side
cv2.imwrite('res.png',res)
