"""Optional visualization for the standalone emotion webcam program."""
import cv2


class FaceMeshOverlay:
    def __init__(self):
        import mediapipe as mp

        self._drawing = mp.solutions.drawing_utils
        self._connections = mp.solutions.face_mesh
        self._mesh = self._connections.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._grid = self._drawing.DrawingSpec(color=(100, 160, 100), thickness=1)
        self._contour = self._drawing.DrawingSpec(color=(0, 230, 255), thickness=1)

    def draw(self, frame):
        results = self._mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for landmarks in results.multi_face_landmarks or []:
            for connections, style in (
                (self._connections.FACEMESH_TESSELATION, self._grid),
                (self._connections.FACEMESH_CONTOURS, self._contour),
            ):
                self._drawing.draw_landmarks(
                    image=frame, landmark_list=landmarks,
                    connections=connections, landmark_drawing_spec=None,
                    connection_drawing_spec=style,
                )

    def close(self):
        self._mesh.close()
