import cv2
import numpy as np


class Segment:
    def __init__(self, id, description, polygon):
        self.id = id
        self.description = description
        self.polygon = polygon  # Lista de (x, y)

    def contains(self, point):
        """
        Verifica si el punto está dentro del polígono del segmento.
        """
        contour = np.array(self.polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(contour, point, False)
        return result >= 0

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data['id'],
            description=data['description'],
            polygon=[tuple(p) for p in data['polygon']]
        )
