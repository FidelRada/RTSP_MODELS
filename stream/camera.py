# stream/camera.py

import cv2
import subprocess
import os
from config import OUTPUT_DIR, FPS
from my_face_recognition.FaceRecognizer import FaceRecognizer

def stream_camera_ffmpeg(rtsp_url, output_path):
    try:
        print(f'{rtsp_url} **********************************************')
        cap = cv2.VideoCapture(rtsp_url)
        print(f"{cap} **********************************************")
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la fuente de video: {rtsp_url}")
            return
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al intentar abrir la fuente de video {rtsp_url}: {e}")
        return

    os.makedirs(output_path, exist_ok=True)

    # Leer el primer frame para obtener la resolución real
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el primer frame de la fuente de video.")
        cap.release()
        return
    height, width = frame.shape[:2]
    print(f"Resolución detectada: {width}x{height}")

    try:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
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

        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        print("[INFO] FFmpeg iniciado")
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al iniciar FFmpeg: {e}")
        cap.release()
        return

    m3u8_path = os.path.join(output_path, "stream.m3u8")

    print(f"📡 Transmitiendo fuente {rtsp_url} en {m3u8_path}")

    # Enviar el primer frame
    try:
        ffmpeg.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al enviar frame a FFmpeg: {e}")
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.wait()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer frame de la fuente de video.")
            break
        # Verifica que la resolución no cambie
        if frame.shape[1] != width or frame.shape[0] != height:
            print(f"[ERROR] La resolución del frame cambió a {frame.shape[1]}x{frame.shape[0]}, se esperaba {width}x{height}")
            break
        try:
            ffmpeg.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"[EXCEPCIÓN] Error al enviar frame a FFmpeg: {e}")
            break
        cv2.imshow("Vista previa", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    cv2.destroyAllWindows()
