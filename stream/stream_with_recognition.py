from my_face_recognition.FaceRecognizer import FaceRecognizer
import cv2
import subprocess
import os
from config import OUTPUT_DIR, FPS

def draw_recognition_results(frame, results):
    for result in results:
        left, top, right, bottom = result["coordenadas"]
        nombre = result["nombre"]
        confianza = result["confianza"]
        # Dibujar rectángulo
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Dibujar texto
        label = f"{nombre} ({confianza:.2f})"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def stream_camera_with_face(rtsp_url, output_path):
    """
    Inicia una transmisión HLS desde una fuente RTSP y realiza reconocimiento facial en cada frame.
    """
    recognizer = FaceRecognizer()
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la fuente de video: {rtsp_url}")
            return
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al intentar abrir la fuente de video {rtsp_url}: {e}")
        return

    os.makedirs(output_path, exist_ok=True)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el primer frame de la fuente de video.")
        cap.release()
        return
    height, width = frame.shape[:2]
    print(f"Resolución detectada: {width}x{height}")

    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "5",
            "-hls_flags", "delete_segments",
            os.path.join(f"{output_path}", "stream.m3u8")
        ]
        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print("[INFO] FFmpeg iniciado para streaming con reconocimiento")
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al iniciar FFmpeg: {e}")
        cap.release()
        return

    # Enviar el primer frame y analizarlo
    try:
        results = recognizer.process_frame(frame)
        frame = draw_recognition_results(frame, results)
        ffmpeg.stdin.write(frame.tobytes())
        print("Primer reconocimiento facial:", results)
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al enviar/procesar el primer frame: {e}")
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.wait()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la transmisión de video.")
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            print(f"[ERROR] La resolución del frame cambió.")
            break
        try:
            # Realizar reconocimiento facial
            results = recognizer.process_frame(frame)
            frame = draw_recognition_results(frame, results)
            # Enviar frame a FFmpeg
            ffmpeg.stdin.write(frame.tobytes())
            if results:
                print("Reconocimiento facial:", results)
        except Exception as e:
            print(f"[EXCEPCIÓN] Error durante el procesamiento del stream: {e}")
            break

    print("[INFO] Limpiando recursos...")
    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("[INFO] Proceso de streaming y reconocimiento finalizado.") 