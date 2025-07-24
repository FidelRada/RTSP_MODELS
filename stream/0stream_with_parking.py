import cv2
import numpy as np
import subprocess
import os
from segment import Segment
from parkingDetector import ParkingDetector

def draw_parking_results(frame, detections, parking_segments, special_segments):
    """
    Dibuja los resultados del parqueo sobre el frame y devuelve el frame procesado
    """
    for det in detections:
        x1, y1, x2, y2 = det['box']
        cx, cy = det['center']
        color = (0, 0, 255) if det['wrongly_parked'] else (0, 255, 0)
        
        # Dibujar rectángulo del auto
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Dibujar centro del auto
        cv2.circle(frame, (cx, cy), 5, color, -1)
        # Etiqueta de información
        label = f"ID:{det['segment_id']} {'Mal' if det['wrongly_parked'] else 'OK'}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    # Dibujar polígonos de segmentos
    for seg in parking_segments:
        cv2.polylines(frame, [np.array(seg.polygon, np.int32).reshape((-1, 1, 2))], 
                     True, (255, 0, 0), 2)
    for seg in special_segments:
        cv2.polylines(frame, [np.array(seg.polygon, np.int32).reshape((-1, 1, 2))], 
                     True, (0, 255, 255), 2)
    
    return frame

def stream_camera_with_parking(rtsp_url, output_folder, parking_segments, special_segments):
    """
    Procesa una fuente RTSP detectando parqueos y transmite la salida procesada
    """
    # Inicializar detector
    detector = ParkingDetector(
        model_path='yolo11n.pt',
        parking_segments=parking_segments,
        special_segments=special_segments
    )
    
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la fuente de video: {rtsp_url}")
            return
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al intentar abrir la fuente de video {rtsp_url}: {e}")
        return

    # Crear directorio de salida
    os.makedirs(output_folder, exist_ok=True)
    
    # Leer primer frame para obtener resolución
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] No se pudo leer el primer frame de la fuente de video.")
        cap.release()
        return
    
    height, width = frame.shape[:2]
    print(f"Resolución detectada: {width}x{height}")
    
    # Comandos de FFmpeg para streaming HLS
    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", "15",  # 15 FPS
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "5",
            "-hls_flags", "delete_segments",
            os.path.join(output_folder, "stream.m3u8")
        ]
        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        print("[INFO] FFmpeg iniciado para streaming con detección de parqueo")
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al iniciar FFmpeg: {e}")
        cap.release()
        return

    # Procesar primer frame
    try:
        detections = detector.detect_cars(frame)
        frame = draw_parking_results(frame, detections, parking_segments, special_segments)
        ffmpeg.stdin.write(frame.tobytes())
        print("Primer análisis de parqueo:", detections)
    except Exception as e:
        print(f"[EXCEPCIÓN] Error al procesar el primer frame: {e}")
        cap.release()
        ffmpeg.stdin.close()
        ffmpeg.wait()
        return

    # Bucle principal de procesamiento
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fin de la transmisión de video.")
            break
            
        try:
            # Detección de autos
            detections = detector.detect_cars(frame)
            # Dibujar resultados
            frame = draw_parking_results(frame, detections)
            # Enviar frame a FFmpeg
            ffmpeg.stdin.write(frame.tobytes())
            
            # Mostrar resultados en ventana (opcional)
            cv2.imshow('Detección de Parqueo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"[EXCEPCIÓN] Error durante el procesamiento del stream: {e}")
            break

    # Liberar recursos
    print("[INFO] Limpiando recursos...")
    cap.release()
    ffmpeg.stdin.close()
    ffmpeg.wait()
    cv2.destroyAllWindows()
    print("[INFO] Proceso de streaming y detección finalizado.")