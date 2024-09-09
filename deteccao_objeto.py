import cv2
import numpy as np

cap = cv2.VideoCapture("/home/mateus/Documentos/VsCode/OpenCV/video.mp4")
thres = 0.5
nms_thres = 0.2

classNames = ["pessoa", "bicicleta", "carro", "motocicleta", "avião",
              "ônibus", "trem", "caminhão", "barco", "semáforo", "hidrante",
              "placa de rua", "sinal de parada", "parquímetro", "banco",
              "pássaro", "gato", "cachorro", "cavalo", "ovelha", "vaca",
              "elefante", "urso", "zebra", "girafa", "chapéu ", "mochila",
              "guarda-chuva", "sapato", "óculos", "bolsa", "gravata", "mala",
              "frisbee", "esquis", "prancha de snowboard", "bola esportiva",
              "pipa", "taco de beisebol ", "luva de beisebol", "skate",
              "prancha de surf", "raquete de tênis", "garrafa", "placa",
              "copo de vinho ", "xícara", "garfo", "faca", "colher", "tigela",
              "banana", "maçã ", "sanduíche", "laranja", "brócolis",
              "cenoura", "cachorro-quente", "pizza", "rosquinha", "bolo",
              "cadeira", "sofá", "planta em vaso", "cama", "espelho",
              "mesa de jantar", "janela", "mesa", "banheiro", "porta", "TV",
              "computador portátil", "rato", "remoto", "teclado","celular",
              "microondas", "forno", "torradeira", "afundar", "frigorífico",
              "liquidificador", "livro", "relógio", "vaso", "tesoura",
              "ursinho de pelúcia", "secador de cabelo", "escova de dentes",
              "escova de cabelo"]

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    ret, video = cap.read()

    video = cv2.flip(video, 1)

    classIds, confis, bbox = net.detect(video, confThreshold = thres)

    bbox = list(bbox)
    confis = list(np.array(confis).reshape(1, -1)[0])
    confis = list(map(float, confis))

    # print(confis)

    indices = cv2.dnn.NMSBoxes(bbox, confis, thres, nms_thres)
    print(indices)

    for i in indices:
        box = bbox[i]
        # print(box)

        x, y, w, h = box[0], box[1], box[2], box[3]

        cv2.rectangle(video, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(video, classNames[classIds[i]-1].upper(), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # print(classIds)
        # print()

    cv2.imshow("Deteccao de Objetos", video)
    key = cv2.waitKey(1) & 0xFF
    if(key == 27):
        break

cv2.destroyAllWindows()