"""Main application window with fully isolated modules: Detection, Face, Gesture"""
import sys
import pathlib
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QGroupBox, QStatusBar,
    QFileDialog, QTabWidget, QMessageBox, QTextEdit, QScrollArea,
    QFormLayout, QLineEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QPixmap, QImage
from src.core.engine import DetectionEngine
from src.utils.helpers import ensure_dir
from loguru import logger

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# ============================================================================
# 🧠 ENGINE STUBS — СОХРАНЯЕМ ИНТЕРФЕЙС DetectionEngine
# ============================================================================

class FaceRecognitionEngine(QThread):
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(float, float, int)  # fps, latency, count
    data_ready = pyqtSignal(list)  # [{"name": "...", "conf": 0.95, "bbox": [...]}, ...]
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._running = False
        self._recording = False
        logger.info(f"👤 FaceEngine init: {config}")

    def run(self):
        self._running = True
        import time, cv2
        cap = cv2.VideoCapture(0 if self.config["source_type"] == "webcam" else self.config["source_path"])
        
        while self._running and cap.isOpened():
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 🔁 ЗАГЛУШКА — замените на вашу реализацию
            faces = [
                {"name": "Иван Петров", "conf": 0.98, "bbox": [120, 80, 220, 200]},
                {"name": "Жест", "conf": 0.92, "bbox": [300, 100, 400, 220]}  # для демонстрации
            ]
            annotated = frame.copy()  # рисуйте рамки/имена здесь

            h, w, ch = annotated.shape
            bytes_per_line = ch * w
            qt_img = QImage(annotated.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            self.frame_ready.emit(qt_img)

            fps = 1.0 / (time.time() - start) if (time.time() - start) > 0 else 0
            self.stats_ready.emit(fps, (time.time()-start)*1000, len(faces))
            self.data_ready.emit(faces)

            if self.config["source_type"] == "image":
                break

        cap.release()
        self._running = False
        self.finished.emit()

    def stop(self):
        self._running = False
        self.wait()

    def toggle_recording(self) -> bool:
        self._recording = not self._recording
        return not self._recording


class GestureRecognitionEngine(QThread):
    frame_ready = pyqtSignal(QImage)
    stats_ready = pyqtSignal(float, float, int)
    data_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._running = False
        self._recording = False
        logger.info(f"✋ GestureEngine init: {config}")

    def run(self):
        self._running = True
        import time, cv2
        cap = cv2.VideoCapture(0 if self.config["source_type"] == "webcam" else self.config["source_path"])

        while self._running and cap.isOpened():
            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 🔁 ЗАГЛУШКА — замените на MediaPipe или вашу модель
            gestures = [
                {"name": "Поднятие руки", "conf": 0.96, "bbox": [250, 120, 350, 240]},
                {"name": "Покачивание головой", "conf": 0.87, "bbox": [400, 90, 500, 210]}
            ]
            annotated = frame.copy()

            h, w, ch = annotated.shape
            bytes_per_line = ch * w
            qt_img = QImage(annotated.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
            self.frame_ready.emit(qt_img)

            fps = 1.0 / (time.time() - start) if (time.time() - start) > 0 else 0
            self.stats_ready.emit(fps, (time.time()-start)*1000, len(gestures))
            self.data_ready.emit(gestures)

            if self.config["source_type"] == "image":
                break

        cap.release()
        self._running = False
        self.finished.emit()

    def stop(self):
        self._running = False
        self.wait()

    def toggle_recording(self) -> bool:
        self._recording = not self._recording
        return not self._recording


# ============================================================================
# 🪟 MAIN WINDOW — ПОЛНОСТЬЮ ИЗОЛИРОВАННЫЕ МОДУЛИ С СВОИМИ НАСТРОЙКАМИ
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO26 Detector PRO — Multi-Module")
        self.setMinimumSize(1280, 850)
        ensure_dir(MODELS_DIR)
        ensure_dir(OUTPUT_DIR)

        # Engines (изолированы)
        self.detection_engine: Optional[DetectionEngine] = None
        self.face_engine: Optional[FaceRecognitionEngine] = None
        self.gesture_engine: Optional[GestureRecognitionEngine] = None

        # Конфиги — каждый модуль имеет свой
        self.det_config = {
            "source_type": "webcam",
            "source_path": "",
            "model_name": "yolo26s.pt",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "conf": 0.25,
            "iou": 0.45,
            "imgsz": 640
        }
        self.face_config = {
            "source_type": "webcam",
            "source_path": "",
            "model_name": "retinaface",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "conf": 0.5,
            "iou": 0.4,
            "imgsz": 640,
            "recognize": True
        }
        self.gesture_config = {
            "source_type": "webcam",
            "source_path": "",
            "model_name": "mediapipe",
            "device": "cpu",
            "conf": 0.5,
            "iou": 0.3,
            "imgsz": 640,
            "mode": "dynamic"  # "dynamic" or "static"
        }

        self.active_module: Optional[str] = None
        self.detection_history: Dict[str, List] = {"detection": [], "face": [], "gesture": []}

        self._init_ui()
        logger.info("🚀 Application started with 3 isolated modules")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; border-radius: 8px; background: #1a1a1a; }
            QTabBar::tab { 
                background: #2a2a2a; color: #aaa; padding: 10px 20px; 
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                margin-right: 2px; min-width: 120px;
            }
            QTabBar::tab:selected { background: #0d6efd; color: white; font-weight: 600; }
            QTabBar::tab:hover { background: #3a3a3a; }
        """)
        
        tabs.addTab(self._create_detection_tab(), "🎯 Detection")
        tabs.addTab(self._create_face_tab(), "👤 Face Recognition")
        tabs.addTab(self._create_gesture_tab(), "✋ Gesture Recognition")

        self.global_status = QStatusBar()
        self.global_status.setStyleSheet("color: #0f0; background: #111; font-weight: 500;")
        self.global_status.showMessage("✅ Ready — Select a module to start")
        self.setStatusBar(self.global_status)

        main_layout.addWidget(tabs)

        # Переключение табов → остановка активного модуля
        tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------------------
    # 🎯 DETECTION TAB — как в оригинале, но с полным контролем конфига
    # ------------------------------------------------------------------------
    def _create_detection_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        # Left: Video
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.det_video = QLabel("Select source and click START")
        self.det_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.det_video.setStyleSheet(
            "background: #0a0a0a; color: #555; border: 2px solid #333; "
            "border-radius: 10px; min-height: 450px; font-size: 14px;"
        )
        left_layout.addWidget(self.det_video)
        self.det_status = QLabel("Idle")
        self.det_status.setStyleSheet("color: #888; font-size: 13px;")
        left_layout.addWidget(self.det_status)
        layout.addWidget(left, 3)

        # Right: Controls + Settings
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)

        # === Source ===
        grp_src = self._group_box("📁 Source")
        lay_src = QVBoxLayout(grp_src)
        self.det_combo_src = QComboBox()
        self.det_combo_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.det_combo_src.currentIndexChanged.connect(self._on_det_source_change)
        lay_src.addWidget(self.det_combo_src)
        self.det_select_file = QPushButton("📂 Select File...")
        self.det_select_file.hide()
        self.det_select_file.clicked.connect(self._select_file_det)
        lay_src.addWidget(self.det_select_file)
        right_layout.addWidget(grp_src)

        # === Model & Device ===
        grp_model = self._group_box("🤖 Model & Device")
        lay_mod = QFormLayout(grp_model)
        self.det_combo_model = QComboBox()
        self.det_combo_model.addItems([
            "yolo26n.pt (Nano)", "yolo26s.pt (Small)",
            "yolo26m.pt (Medium)", "yolo26l.pt (Large)"
        ])
        self.det_combo_model.setCurrentText("yolo26s.pt (Small)")
        lay_mod.addRow("Model:", self.det_combo_model)

        self.det_combo_device = QComboBox()
        self.det_combo_device.addItems(["CPU", "CUDA"])
        if torch.cuda.is_available():
            self.det_combo_device.setCurrentIndex(1)
        lay_mod.addRow("Device:", self.det_combo_device)
        right_layout.addWidget(grp_model)

        # === Thresholds ===
        grp_thr = self._group_box("⚙️ Thresholds")
        lay_thr = QFormLayout(grp_thr)

        self.det_slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.det_slider_conf.setRange(5, 95)
        self.det_slider_conf.setValue(25)
        self.det_lbl_conf = QLabel("0.25")
        self.det_slider_conf.valueChanged.connect(
            lambda v: self.det_lbl_conf.setText(f"{v/100:.2f}")
        )
        lay_thr.addRow("Confidence:", self._slider_with_label(self.det_slider_conf, self.det_lbl_conf))

        self.det_slider_iou = QSlider(Qt.Orientation.Horizontal)
        self.det_slider_iou.setRange(5, 95)
        self.det_slider_iou.setValue(45)
        self.det_lbl_iou = QLabel("0.45")
        self.det_slider_iou.valueChanged.connect(
            lambda v: self.det_lbl_iou.setText(f"{v/100:.2f}")
        )
        lay_thr.addRow("IoU (NMS):", self._slider_with_label(self.det_slider_iou, self.det_lbl_iou))

        self.det_spin_imgsz = QLineEdit("640")
        self.det_spin_imgsz.setValidator(None)  # можно добавить QIntValidator
        lay_thr.addRow("Input Size:", self.det_spin_imgsz)
        right_layout.addWidget(grp_thr)

        # === Control ===
        grp_ctrl = self._group_box("🎮 Control")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        btn_row = QHBoxLayout()
        self.det_start = QPushButton("▶ START")
        self.det_start.setStyleSheet(self._btn_style("#2ecc71"))
        self.det_start.clicked.connect(lambda: self._start_module("detection"))
        btn_row.addWidget(self.det_start)

        self.det_stop = QPushButton("⏹ STOP")
        self.det_stop.setEnabled(False)
        self.det_stop.setStyleSheet(self._btn_style("#e74c3c"))
        self.det_stop.clicked.connect(lambda: self._stop_module("detection"))
        btn_row.addWidget(self.det_stop)
        lay_ctrl.addLayout(btn_row)
        right_layout.addWidget(grp_ctrl)

        # === Export ===
        grp_exp = self._group_box("💾 Export")
        lay_exp = QHBoxLayout(grp_exp)
        self.det_rec = QPushButton("🔴 REC")
        self.det_rec.setStyleSheet(self._btn_style("#333"))
        self.det_rec.clicked.connect(lambda: self._toggle_rec("detection"))
        lay_exp.addWidget(self.det_rec)

        self.det_shot = QPushButton("📸 SC")
        self.det_shot.setStyleSheet(self._btn_style("#333"))
        self.det_shot.clicked.connect(lambda: self._take_screenshot(self.det_video, "det"))
        lay_exp.addWidget(self.det_shot)

        self.det_json = QPushButton("📝 JSON")
        self.det_json.setStyleSheet(self._btn_style("#333"))
        self.det_json.clicked.connect(lambda: self._save_json("detection"))
        lay_exp.addWidget(self.det_json)
        right_layout.addWidget(grp_exp)

        # === Results ===
        grp_res = self._group_box("🎯 Detected Objects")
        lay_res = QVBoxLayout(grp_res)
        self.det_results = QTextEdit()
        self.det_results.setReadOnly(True)
        self.det_results.setStyleSheet(
            "background: #111; color: #0f0; font-family: Consolas; font-size: 11px;"
        )
        lay_res.addWidget(self.det_results)
        right_layout.addWidget(grp_res)

        right_layout.addStretch()
        layout.addWidget(right, 1)
        return tab

    # ------------------------------------------------------------------------
    # 👤 FACE TAB — СВОИ НАСТРОЙКИ
    # ------------------------------------------------------------------------
    def _create_face_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.face_video = QLabel("Face module ready — click START")
        self.face_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_video.setStyleSheet(
            "background: #0a0a0a; color: #555; border: 2px solid #333; "
            "border-radius: 10px; min-height: 450px; font-size: 14px;"
        )
        left_layout.addWidget(self.face_video)
        self.face_status = QLabel("Idle")
        self.face_status.setStyleSheet("color: #888; font-size: 13px;")
        left_layout.addWidget(self.face_status)
        layout.addWidget(left, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)

        # Source
        grp_src = self._group_box("📁 Source")
        lay_src = QVBoxLayout(grp_src)
        self.face_combo_src = QComboBox()
        self.face_combo_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.face_combo_src.currentIndexChanged.connect(self._on_face_source_change)
        lay_src.addWidget(self.face_combo_src)
        self.face_select_file = QPushButton("📂 Select File...")
        self.face_select_file.hide()
        self.face_select_file.clicked.connect(self._select_file_face)
        lay_src.addWidget(self.face_select_file)
        right_layout.addWidget(grp_src)

        # Model & Options
        grp_model = self._group_box("🤖 Face Model")
        lay_mod = QFormLayout(grp_model)
        self.face_combo_model = QComboBox()
        self.face_combo_model.addItems([
            "retinaface", "mtcnn", "yoloface", "insightface"
        ])
        self.face_combo_model.setCurrentText("retinaface")
        lay_mod.addRow("Model:", self.face_combo_model)

        self.face_combo_device = QComboBox()
        self.face_combo_device.addItems(["CPU", "CUDA"])
        if torch.cuda.is_available():
            self.face_combo_device.setCurrentIndex(1)
        lay_mod.addRow("Device:", self.face_combo_device)

        self.face_chk_recognize = QCheckBox("🔍 Recognize names (requires DB)")
        self.face_chk_recognize.setChecked(True)
        lay_mod.addRow("", self.face_chk_recognize)
        right_layout.addWidget(grp_model)

        # Thresholds
        grp_thr = self._group_box("⚙️ Thresholds")
        lay_thr = QFormLayout(grp_thr)
        self.face_slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.face_slider_conf.setRange(5, 95)
        self.face_slider_conf.setValue(50)
        self.face_lbl_conf = QLabel("0.50")
        self.face_slider_conf.valueChanged.connect(
            lambda v: self.face_lbl_conf.setText(f"{v/100:.2f}")
        )
        lay_thr.addRow("Confidence:", self._slider_with_label(self.face_slider_conf, self.face_lbl_conf))

        self.face_spin_imgsz = QLineEdit("640")
        lay_thr.addRow("Input Size:", self.face_spin_imgsz)
        right_layout.addWidget(grp_thr)

        # Control
        grp_ctrl = self._group_box("🎮 Control")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        btn_row = QHBoxLayout()
        self.face_start = QPushButton("▶ START")
        self.face_start.setStyleSheet(self._btn_style("#2ecc71"))
        self.face_start.clicked.connect(lambda: self._start_module("face"))
        btn_row.addWidget(self.face_start)

        self.face_stop = QPushButton("⏹ STOP")
        self.face_stop.setEnabled(False)
        self.face_stop.setStyleSheet(self._btn_style("#e74c3c"))
        self.face_stop.clicked.connect(lambda: self._stop_module("face"))
        btn_row.addWidget(self.face_stop)
        lay_ctrl.addLayout(btn_row)
        right_layout.addWidget(grp_ctrl)

        # Export
        grp_exp = self._group_box("💾 Export")
        lay_exp = QHBoxLayout(grp_exp)
        self.face_rec = QPushButton("🔴 REC")
        self.face_rec.setStyleSheet(self._btn_style("#333"))
        self.face_rec.clicked.connect(lambda: self._toggle_rec("face"))
        lay_exp.addWidget(self.face_rec)

        self.face_shot = QPushButton("📸 SC")
        self.face_shot.setStyleSheet(self._btn_style("#333"))
        self.face_shot.clicked.connect(lambda: self._take_screenshot(self.face_video, "face"))
        lay_exp.addWidget(self.face_shot)

        self.face_json = QPushButton("📝 JSON")
        self.face_json.setStyleSheet(self._btn_style("#333"))
        self.face_json.clicked.connect(lambda: self._save_json("face"))
        lay_exp.addWidget(self.face_json)
        right_layout.addWidget(grp_exp)

        # Results
        grp_res = self._group_box("👥 Detected Faces")
        lay_res = QVBoxLayout(grp_res)
        self.face_results = QTextEdit()
        self.face_results.setReadOnly(True)
        self.face_results.setStyleSheet(
            "background: #111; color: #0ff; font-family: Consolas; font-size: 11px;"
        )
        lay_res.addWidget(self.face_results)
        right_layout.addWidget(grp_res)

        right_layout.addStretch()
        layout.addWidget(right, 1)
        return tab

    # ------------------------------------------------------------------------
    # ✋ GESTURE TAB — СВОИ НАСТРОЙКИ
    # ------------------------------------------------------------------------
    def _create_gesture_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.gesture_video = QLabel("Gesture module ready — click START")
        self.gesture_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gesture_video.setStyleSheet(
            "background: #0a0a0a; color: #555; border: 2px solid #333; "
            "border-radius: 10px; min-height: 450px; font-size: 14px;"
        )
        left_layout.addWidget(self.gesture_video)
        self.gesture_status = QLabel("Idle")
        self.gesture_status.setStyleSheet("color: #888; font-size: 13px;")
        left_layout.addWidget(self.gesture_status)
        layout.addWidget(left, 3)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)

        # Source
        grp_src = self._group_box("📁 Source")
        lay_src = QVBoxLayout(grp_src)
        self.gesture_combo_src = QComboBox()
        self.gesture_combo_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.gesture_combo_src.currentIndexChanged.connect(self._on_gesture_source_change)
        lay_src.addWidget(self.gesture_combo_src)
        self.gesture_select_file = QPushButton("📂 Select File...")
        self.gesture_select_file.hide()
        self.gesture_select_file.clicked.connect(self._select_file_gesture)
        lay_src.addWidget(self.gesture_select_file)
        right_layout.addWidget(grp_src)

        # Model & Mode
        grp_model = self._group_box("🤖 Gesture Model")
        lay_mod = QFormLayout(grp_model)
        self.gesture_combo_model = QComboBox()
        self.gesture_combo_model.addItems([
            "mediapipe", "custom_cnn", "handnet"
        ])
        self.gesture_combo_model.setCurrentText("mediapipe")
        lay_mod.addRow("Model:", self.gesture_combo_model)

        self.gesture_combo_device = QComboBox()
        self.gesture_combo_device.addItems(["CPU", "CUDA"])
        self.gesture_combo_device.setCurrentIndex(0)  # default CPU for hands
        lay_mod.addRow("Device:", self.gesture_combo_device)

        self.gesture_combo_mode = QComboBox()
        self.gesture_combo_mode.addItems(["⚡ Dynamic (Video)", "🎯 Static (Image)"])
        lay_mod.addRow("Mode:", self.gesture_combo_mode)
        right_layout.addWidget(grp_model)

        # Thresholds
        grp_thr = self._group_box("⚙️ Thresholds")
        lay_thr = QFormLayout(grp_thr)
        self.gesture_slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.gesture_slider_conf.setRange(5, 95)
        self.gesture_slider_conf.setValue(50)
        self.gesture_lbl_conf = QLabel("0.50")
        self.gesture_slider_conf.valueChanged.connect(
            lambda v: self.gesture_lbl_conf.setText(f"{v/100:.2f}")
        )
        lay_thr.addRow("Confidence:", self._slider_with_label(self.gesture_slider_conf, self.gesture_lbl_conf))

        self.gesture_spin_imgsz = QLineEdit("640")
        lay_thr.addRow("Input Size:", self.gesture_spin_imgsz)
        right_layout.addWidget(grp_thr)

        # Control
        grp_ctrl = self._group_box("🎮 Control")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        btn_row = QHBoxLayout()
        self.gesture_start = QPushButton("▶ START")
        self.gesture_start.setStyleSheet(self._btn_style("#2ecc71"))
        self.gesture_start.clicked.connect(lambda: self._start_module("gesture"))
        btn_row.addWidget(self.gesture_start)

        self.gesture_stop = QPushButton("⏹ STOP")
        self.gesture_stop.setEnabled(False)
        self.gesture_stop.setStyleSheet(self._btn_style("#e74c3c"))
        self.gesture_stop.clicked.connect(lambda: self._stop_module("gesture"))
        btn_row.addWidget(self.gesture_stop)
        lay_ctrl.addLayout(btn_row)
        right_layout.addWidget(grp_ctrl)

        # Export
        grp_exp = self._group_box("💾 Export")
        lay_exp = QHBoxLayout(grp_exp)
        self.gesture_rec = QPushButton("🔴 REC")
        self.gesture_rec.setStyleSheet(self._btn_style("#333"))
        self.gesture_rec.clicked.connect(lambda: self._toggle_rec("gesture"))
        lay_exp.addWidget(self.gesture_rec)

        self.gesture_shot = QPushButton("📸 SC")
        self.gesture_shot.setStyleSheet(self._btn_style("#333"))
        self.gesture_shot.clicked.connect(lambda: self._take_screenshot(self.gesture_video, "gesture"))
        lay_exp.addWidget(self.gesture_shot)

        self.gesture_json = QPushButton("📝 JSON")
        self.gesture_json.setStyleSheet(self._btn_style("#333"))
        self.gesture_json.clicked.connect(lambda: self._save_json("gesture"))
        lay_exp.addWidget(self.gesture_json)
        right_layout.addWidget(grp_exp)

        # Results
        grp_res = self._group_box("✋ Detected Gestures")
        lay_res = QVBoxLayout(grp_res)
        self.gesture_results = QTextEdit()
        self.gesture_results.setReadOnly(True)
        self.gesture_results.setStyleSheet(
            "background: #111; color: #f0f; font-family: Consolas; font-size: 11px;"
        )
        lay_res.addWidget(self.gesture_results)
        right_layout.addWidget(grp_res)

        right_layout.addStretch()
        layout.addWidget(right, 1)
        return tab

    # ------------------------------------------------------------------------
    # 🛠️ УТИЛИТЫ
    # ------------------------------------------------------------------------
    def _group_box(self, title: str) -> QGroupBox:
        gb = QGroupBox(title)
        gb.setStyleSheet("""
            QGroupBox {
                font-weight: 600; color: #fff; border: 1px solid #444;
                border-radius: 8px; margin-top: 8px; padding-top: 12px;
                background: #1e1e1e;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        """)
        return gb

    def _btn_style(self, bg: str) -> str:
        return f"""
            QPushButton {{
                background: {bg}; color: white; border: none;
                border-radius: 6px; padding: 8px 16px; font-weight: 500;
            }}
            QPushButton:hover {{ background: {bg}cc; }}
            QPushButton:disabled {{ background: #444; color: #888; }}
        """

    def _slider_with_label(self, slider: QSlider, label: QLabel) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(slider)
        lay.addWidget(label)
        return w

    # ------------------------------------------------------------------------
    # 📥 SOURCE HANDLERS
    # ------------------------------------------------------------------------
    def _on_det_source_change(self, idx: int):
        self.det_select_file.setVisible(idx > 0)

    def _on_face_source_change(self, idx: int):
        self.face_select_file.setVisible(idx > 0)

    def _on_gesture_source_change(self, idx: int):
        self.gesture_select_file.setVisible(idx > 0)

    def _select_file_det(self):
        self._select_file(self.det_combo_src, "det")

    def _select_file_face(self):
        self._select_file(self.face_combo_src, "face")

    def _select_file_gesture(self):
        self._select_file(self.gesture_combo_src, "gesture")

    def _select_file(self, combo: QComboBox, prefix: str):
        idx = combo.currentIndex()
        if idx == 0: return
        title = ["", "Video", "Image"][idx]
        ext = ["", "Videos (*.mp4 *.avi *.mkv)", "Images (*.jpg *.png)"][idx]
        file, _ = QFileDialog.getOpenFileName(self, f"Select {title}", "", ext)
        if file:
            self.source_path = file
            getattr(self, f"{prefix}_status").setText(f"📄 {pathlib.Path(file).name}")

    # ------------------------------------------------------------------------
    # ⚙️ START / STOP MODULES — С ОБНОВЛЕНИЕМ КОНФИГА
    # ------------------------------------------------------------------------
    def _start_module(self, module: str):
        if self.active_module and self.active_module != module:
            self._stop_module(self.active_module)

        try:
            # Обновляем конфиг из UI
            if module == "detection":
                self.det_config.update({
                    "source_type": ["webcam", "video", "image"][self.det_combo_src.currentIndex()],
                    "source_path": self.source_path,
                    "model_name": self.det_combo_model.currentText().split()[0],
                    "device": self.det_combo_device.currentText().lower(),
                    "conf": self.det_slider_conf.value() / 100.0,
                    "iou": self.det_slider_iou.value() / 100.0,
                    "imgsz": int(self.det_spin_imgsz.text() or 640)
                })
                engine = self.detection_engine = DetectionEngine(**self.det_config)
                video_label = self.det_video
                status_label = self.det_status
                start_btn, stop_btn = self.det_start, self.det_stop
                results_widget = self.det_results

            elif module == "face":
                self.face_config.update({
                    "source_type": ["webcam", "video", "image"][self.face_combo_src.currentIndex()],
                    "source_path": self.source_path,
                    "model_name": self.face_combo_model.currentText(),
                    "device": self.face_combo_device.currentText().lower(),
                    "conf": self.face_slider_conf.value() / 100.0,
                    "iou": 0.4,  # не используется в face, но оставим для совместимости
                    "imgsz": int(self.face_spin_imgsz.text() or 640),
                    "recognize": self.face_chk_recognize.isChecked()
                })
                engine = self.face_engine = FaceRecognitionEngine(self.face_config)
                video_label = self.face_video
                status_label = self.face_status
                start_btn, stop_btn = self.face_start, self.face_stop
                results_widget = self.face_results

            else:  # gesture
                self.gesture_config.update({
                    "source_type": ["webcam", "video", "image"][self.gesture_combo_src.currentIndex()],
                    "source_path": self.source_path,
                    "model_name": self.gesture_combo_model.currentText(),
                    "device": self.gesture_combo_device.currentText().lower(),
                    "conf": self.gesture_slider_conf.value() / 100.0,
                    "iou": 0.3,
                    "imgsz": int(self.gesture_spin_imgsz.text() or 640),
                    "mode": "dynamic" if self.gesture_combo_mode.currentIndex() == 0 else "static"
                })
                engine = self.gesture_engine = GestureRecognitionEngine(self.gesture_config)
                video_label = self.gesture_video
                status_label = self.gesture_status
                start_btn, stop_btn = self.gesture_start, self.gesture_stop
                results_widget = self.gesture_results

            # Сброс данных
            self.detection_history[module].clear()
            results_widget.clear()

            # Подписка
            engine.frame_ready.connect(lambda img: self._update_frame(img, video_label))
            engine.stats_ready.connect(lambda fps, lat, cnt: self._update_stats(fps, lat, cnt, status_label, module))
            engine.data_ready.connect(lambda data: self._update_results(data, results_widget, module))
            engine.error_occurred.connect(lambda err: self._handle_error(err, module))
            engine.finished.connect(lambda: self._on_engine_finished(module))

            engine.start()
            self.active_module = module

            start_btn.setEnabled(False)
            stop_btn.setEnabled(True)
            status_label.setText("⚡ Processing...")
            self.global_status.showMessage(f"🔥 {module.upper()} module running")

        except Exception as e:
            logger.error(f"❌ Start failed ({module}): {e}")
            QMessageBox.critical(self, "Error", str(e))
            self._stop_module(module)

    def _stop_module(self, module: str):
        if module == "detection" and self.detection_engine and self.detection_engine.isRunning():
            self.detection_engine.stop()
        elif module == "face" and self.face_engine and self.face_engine.isRunning():
            self.face_engine.stop()
        elif module == "gesture" and self.gesture_engine and self.gesture_engine.isRunning():
            self.gesture_engine.stop()

        # UI reset
        prefix = "det" if module == "detection" else module
        start_btn = getattr(self, f"{prefix}_start")
        stop_btn = getattr(self, f"{prefix}_stop")
        status_label = getattr(self, f"{prefix}_status")
        rec_btn = getattr(self, f"{prefix}_rec")

        start_btn.setEnabled(True)
        stop_btn.setEnabled(False)
        status_label.setText("⏹️ Stopped")
        if rec_btn.text() == "⏹ STOP":
            rec_btn.setText("🔴 REC")
            rec_btn.setStyleSheet(self._btn_style("#333"))

        if self.active_module == module:
            self.active_module = None
            self.global_status.showMessage("✅ Module stopped — Ready")

    def _on_engine_finished(self, module: str):
        prefix = "det" if module == "detection" else module
        start_btn = getattr(self, f"{prefix}_start")
        stop_btn = getattr(self, f"{prefix}_stop")
        status_label = getattr(self, f"{prefix}_status")

        start_btn.setEnabled(True)
        stop_btn.setEnabled(False)
        if status_label.text() == "⚡ Processing...":
            status_label.setText("✅ Finished")

        if self.active_module == module:
            self.active_module = None

    def _handle_error(self, error_msg: str, module: str):
        QMessageBox.critical(self, f"{module.capitalize()} Error", error_msg)
        self._stop_module(module)

    def _toggle_rec(self, module: str):
        prefix = "det" if module == "detection" else module
        engine = getattr(self, f"{module}_engine")
        btn = getattr(self, f"{prefix}_rec")
        if engine and engine.isRunning():
            is_stopped = engine.toggle_recording()
            if is_stopped:
                btn.setText("🔴 REC")
                btn.setStyleSheet(self._btn_style("#333"))
                QMessageBox.information(self, "Ready", f"Video saved to /output/{module}/")
            else:
                btn.setText("⏹ STOP")
                btn.setStyleSheet(self._btn_style("#e74c3c"))

    def _take_screenshot(self, video_label: QLabel, prefix: str):
        pix = video_label.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if pix.save(str(path)):
                self.global_status.showMessage(f"📸 Saved: {path.name}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save screenshot")

    def _save_json(self, module: str):
        data = self.detection_history[module]
        if not data:
            QMessageBox.warning(self, "No Data", "No detections to export")
            return
        try:
            path = OUTPUT_DIR / f"{module}_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", f"Report saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    # ------------------------------------------------------------------------
    # 📡 SIGNAL HANDLERS
    # ------------------------------------------------------------------------
    def _update_frame(self, q_img: QImage, video_label: QLabel):
        pix = QPixmap.fromImage(q_img)
        video_label.setPixmap(pix.scaled(
            video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def _update_stats(self, fps: float, latency: float, count: int, status_label: QLabel, module: str):
        status_label.setText(f"FPS: {fps:.1f} | Lat: {latency:.1f}ms | {count}")

    def _update_results(self, data: List[Dict], results_widget: QTextEdit, module: str):
        self.detection_history[module].extend(data[:1000])  # ограничение

        if not data:
            results_widget.clear()
            return

        lines = []
        for item in data[:10]:  # макс. 10 строк
            name = item.get("name", item.get("class_name", "unknown"))
            conf = item.get("conf", item.get("confidence", 1.0))
            bbox = item.get("bbox", [])
            line = f"• {name} ({conf:.0%}) [{bbox[:2] if bbox else ''}]"
            lines.append(line)

        results_widget.setText("\n".join(lines))

    def _on_tab_changed(self, index: int):
        modules = ["detection", "face", "gesture"]
        new_module = modules[index]
        if self.active_module and self.active_module != new_module:
            self._stop_module(self.active_module)

    def closeEvent(self, event):
        for mod in ["detection", "face", "gesture"]:
            self._stop_module(mod)
        logger.info("👋 Application closed")
        event.accept()


# ============================================================================
# 🚀 RUN
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        * { font-family: 'Segoe UI', sans-serif; }
        QMainWindow, QWidget { background: #1a1a1a; color: #eee; }
        QLabel { color: #ddd; }
        QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 8px; }
        QComboBox, QLineEdit, QSlider::groove {
            background: #2a2a2a; border: 1px solid #444; border-radius: 4px;
            color: #fff; padding: 4px;
        }
        QPushButton {
            border: none; border-radius: 5px; padding: 6px 12px;
        }
        QTextEdit {
            background: #111; border: 1px solid #333; border-radius: 4px;
            font-family: Consolas; font-size: 11px;
        }
        QScrollBar {
            background: #222; width: 10px;
        }
        QScrollBar::handle { background: #555; border-radius: 4px; }
    """)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()