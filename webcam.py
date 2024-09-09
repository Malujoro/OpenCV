import cv2

captura = cv2.VideoCapture(0)
largura = 420
altura = 320

while True:
    retorno, video = captura.read()

    videoRed = cv2.resize(video, (largura, altura), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(videoRed, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Video", videoRed)
    cv2.imshow("VideoGray", gray)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        break

captura.release()
cv2.destroyAllWindows()