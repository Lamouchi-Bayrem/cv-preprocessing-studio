from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen



class ImageWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.start_point = None
        self.end_point = None
        self.roi_rect = None

    def mousePressEvent(self, event):
        self.start_point = event.position().toPoint()
        self.end_point = self.start_point
        self.update()

    def mouseMoveEvent(self, event):
        if self.start_point:
            self.end_point = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        self.end_point = event.position().toPoint()
        self.roi_rect = QRect(self.start_point, self.end_point)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.start_point and self.end_point:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.GlobalColor.red, 2))
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)

    def get_roi_coords(self):
        if not self.roi_rect:
            return None
        r = self.roi_rect.normalized()
        return r.x(), r.y(), r.x() + r.width(), r.y() + r.height()
