import cv2

# Inicializa o
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, video = cap.read()

    # Espelha o vídeo
    video = cv2.flip(video, 1)

    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)

    # Baseado no algoritmo dado, o computador faz algumas detecções
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize = (100, 100))

    # Cria um retângulo nos pontos de coordenadas X e Y
    # Também utiliza Largura e Altura
    for(x, y, w, h) in faces:
        print(f"Face: {w} {h}")
        cv2.rectangle(video, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # "Pega" a matriz que possui a face detectada
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = video[y:y+h, x:x+w]

        print(int(x+w/2), int(y+h/2))

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for(x_eye, y_eye, w_eye, h_eye) in eyes:
            cv2.rectangle(roi_color, (x_eye, y_eye), (x_eye+w_eye, y_eye+h_eye), (0, 0, 255), 2)

    cv2.imshow("Reconhecimento Facial", video)
    key = cv2.waitKey(1) & 0xFF

    # Tecla ESC
    if(key == 27):
        break

cap.release
cv2.destroyAllWindows()