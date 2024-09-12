import cv2

# rastreador = cv2.TrackerKCF_create()
rastreador = cv2.TrackerCSRT_create()

# video = cv2.VideoCapture("Rastreamento/race.mp4")
video = cv2.VideoCapture("Rastreamento/street.mp4")
ret, frame = video.read()

# Regio = Region of Interest
bbox = cv2.selectROI(frame)
# print(bbox)

ok = rastreador.init(frame, bbox)
# print(ok)

while True:

    ok, frame = video.read()
    # print(ok)

    if(not ok):
        break

    ok, bbox = rastreador.update(frame)
    # print(bbox)

    if(ok):
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        cv2.putText(frame, "Erro", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Rastreamento", frame)

    if(cv2.waitKey(1) & 0xFF == 27):
        break

video.release()
cv2.destroyAllWindows()