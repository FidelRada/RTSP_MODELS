# stream/streamer.py

import cv2
import os
import time
from .camera import get_camera
from .ffmpeg_writer import start_ffmpeg

def start_stream(camera_id, output_dir):
    cap = get_camera(camera_id)
    if cap is None:
        print(f"[ERROR] Streaming abortado: no se pudo abrir la cámara {camera_id}")
        return
    ffmpeg = start_ffmpeg(f'{output_dir}')

    # Esperar a que se cree el archivo .m3u8
    m3u8_path = os.path.join(output_dir, "stream.m3u8")
    while not os.path.exists(m3u8_path):
        print(f"🕓 Esperando a que FFmpeg cree {m3u8_path}...")
        time.sleep(0.5)

    print(f"📡 Transmitiendo cámara {camera_id} en {m3u8_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ffmpeg.stdin.write(frame.tobytes())

        # Mostrar vista previa
        cv2.imshow("Vista previa", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    cv2.destroyAllWindows()
