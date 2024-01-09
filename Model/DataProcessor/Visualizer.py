import cv2


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def draw_bounding_box(frame, box_of_interest, label=""):
        # Skopiuj obraz, aby uniknąć modyfikacji oryginalnego obrazu
        annotated_frame = frame.copy()

        # Wyciągnij współrzędne bounding boxa
        x1, y1, x2, y2 = box_of_interest.numpy()

        # Przekształć współrzędne na liczby całkowite
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Narysuj bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Dodaj label, jeśli został dostarczony
        if label:
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated_frame
