import cv2
import numpy as np
from typing import *
from .segment import Segment
from ultralytics import YOLO

class ParkingDetector:
    """
    Detector de autos para análisis de estacionamiento usando YOLO.

    Atributos:
        model: Modelo YOLO cargado.
        parking_segments: lista de segmentos de parqueo (Segment).
        special_segments: lista de segmentos especiales (Segment).
    """
    def __init__(
        self,
        model_path: str,
        parking_segments: List[Segment],
        special_segments: List[Segment]
    ):
        self.model = YOLO(model_path)
        self.parking_segments = parking_segments
        self.special_segments = special_segments

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Carga una imagen desde archivo."""
        return cv2.imread(image_path)

    def detect_cars(self, image: np.ndarray) -> List[dict]:
        """
        Detecta autos en la imagen, evalúa su posición y retorna información detallada.

        Retorna:
            List[dict] con claves:
                - box: tupla (x1, y1, x2, y2)
                - center: tupla (cx, cy)
                - confidence: float
                - segment_id: str o None
                - segment_description: str o None
                - wrongly_parked: bool
        """
        results = self.model(image)
        detections: List[dict] = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != 2:
                    continue  # Ignorar clases distintas de auto

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                center = (cx, cy)

                # Chequear si está en zona especial
                seg_special = next(
                    (seg for seg in self.special_segments if seg.contains(center)),
                    None
                )
                if seg_special:
                    segment_id = seg_special.id
                    segment_desc = seg_special.description
                    wrongly = False  # no evaluamos parqueo en zona especial
                else:
                    # Chequear parqueo válido
                    seg_park = next(
                        (seg for seg in self.parking_segments if seg.contains(center)),
                        None
                    )
                    if seg_park:
                        segment_id = seg_park.id
                        segment_desc = seg_park.description
                        wrongly = False
                    else:
                        segment_id = None
                        segment_desc = None
                        wrongly = True

                detections.append({
                    'box': (x1, y1, x2, y2),
                    'center': center,
                    'confidence': conf,
                    'segment_id': segment_id,
                    'segment_description': segment_desc,
                    'wrongly_parked': wrongly
                })

        return detections

    def draw_results(self, image: np.ndarray, detections: List[dict]) -> None:
        """
        Dibuja cuadros y anotaciones sobre los autos detectados en la imagen.
        """
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cx, cy = det['center']
            color = (0, 0, 255) if det['wrongly_parked'] else (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(image, (cx, cy), 5, color, -1)
            label = (
                f"ID:{det['segment_id']} "
                f"{'Mal' if det['wrongly_parked'] else 'OK'}"
            )
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Dibujar polígonos de segmentos
        for seg in self.parking_segments:
            pts = np.array(seg.polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (255, 0, 0), 2)
        for seg in self.special_segments:
            pts = np.array(seg.polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 255), 2)

        cv2.imshow('Resultado', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Ejemplo de uso
if __name__ == '__main__':
    parking_segments = [
        Segment('P1', 'Parqueo Frente', [(100, 200), (200, 150), (300, 200), (400, 500)]),
        Segment('P2', 'Parqueo Centro', [(450, 200), (550, 150), (650, 200), (750, 500)]),
        Segment('P3', 'Parqueo Trasero', [(800, 200), (900, 150), (1000, 200), (1100, 500)])
    ]

    print(parking_segments)

    special_segments = [
        Segment('S1', 'Zona Especial', [(30, 30), (1100, 30), (1100, 190), (30, 190)])
    ]

    detector = ParkingDetector(
        model_path='yolo11n.pt',
        parking_segments=parking_segments,
        special_segments=special_segments
    )

    img = detector.load_image('Images/imagen_0005.jpg')
    detections = detector.detect_cars(img)
    print("asadasdadadadadsadsdaasdadasdas")
    print(f"mmmmmmmmm: {detections}")
    detector.draw_results(img, detections)
