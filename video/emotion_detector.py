import cv2
from fer import FER

# Inicialize a webcam
cap = cv2.VideoCapture(0)

# Crie o detector de emoções
detector = FER(mtcnn=True)  # Use o MTCNN para uma melhor detecção facial

while True:
    # Capture o quadro da webcam
    ret, frame = cap.read()

    # Detecte emoções no quadro
    emotions = detector.detect_emotions(frame)

    # Se detectar uma face, desenhe um retângulo ao redor dela e mostre as emoções
    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostre a emoção provável
        dominant_emotion = max(emotion["emotions"], key=emotion["emotions"].get)
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Exiba o quadro
    cv2.imshow('Emotional Detector', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a webcam e feche as janelas
cap.release()
cv2.destroyAllWindows()
