import cv2

# Ruta o URL del video/RTSP
source = "rtsp://localhost:8554/videoRTSP"

cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("❌ No se puede abrir la fuente.")
    exit()

# Configurar ventana
win_name = "▶️ Reproductor RTSP - OpenCV"
cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Fin del video o conexión perdida.")
        break

    cv2.imshow(win_name, frame)

    # Control de velocidad: ~30 FPS
    key = cv2.waitKey(30) & 0xFF  # 30ms ≈ 33 FPS

    if key == ord('q') or key == 27:  # 'q' o Esc
        break
    elif key == ord(' '):  # Espacio para pausa
        print("⏸️ Pausa (presiona cualquier tecla para continuar)")
        cv2.waitKey(0)  # Espera hasta que presiones una tecla

cap.release()
cv2.destroyAllWindows()