import cv2
oldPath = './old/'
newPath = './new/'

for i in range(1, 5):
	image = cv2.imread(oldPath + str(i) + '.jpg')
	cv2.imwrite(newPath + str(i) + '.jpg', image[image.shape[0] // 4: image.shape[0] // 4 * 3, image.shape[1] // 4: image.shape[1] // 4 * 3])