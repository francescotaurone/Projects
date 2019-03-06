import cv2


vidcap = cv2.VideoCapture('Elisa.mp4')
success, image = vidcap.read()
count = 0
success = True
while success:
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    cv2.imwrite("./images_video/8_%d.jpg" % count, image)     # save frame as JPEG file
    count += 1
