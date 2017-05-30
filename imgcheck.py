import cv2
path = "D:\\map_data\\1496105141199\\1496105151509.png"
img = cv2.imread(path,0)
w = int(img.shape[0]/2)
h = int(img.shape[1]/2)
s_img = img[w:h, 50:50]
x_offset=y_offset=50
edges = cv2.Canny(img,10,50)
#edges[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
img = img
cv2.imwrite('bw_img.png', edges)
cv2.imwrite('img.png', img)
