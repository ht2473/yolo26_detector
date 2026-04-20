"""Main application window"""
import sys
import pathlib
import json
from datetime import datetime
from typing import Optional, List
import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton,
                             QComboBox, QSlider, QGroupBox, QStatusBar,
                             QFileDialog, QTabWidget, QMessageBox, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from src.core.engine import DetectionEngine
from src.utils.helpers import ensure_dir  # Использование хелперов
from loguru import logger

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO26 Detector PRO")
        self.setMinimumSize(1200, 800)
        
        ensure_dir(MODELS_DIR)
        ensure_dir(OUTPUT_DIR)
        
        self.engine: Optional[DetectionEngine] = None
        self.source_path: str = ""
        self.detection_history: List = []
        self.fps_history: List = []
        
        self._init_ui()
        logger.info("🚀 Application started")

    def _init_ui(self) -> None:
        """Initialize user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { height: 30px; padding: 5px 15px; }")
        
        # === TAB 1: DETECTION ===
        tab_detection = QWidget()
        det_layout = QHBoxLayout(tab_detection)
        
        # Left panel: Video display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.video_label = QLabel("Select source and click START")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet(
            "background: #111; color: #555; border-radius: 8px; min-height: 480px;"
        )
        left_layout.addWidget(self.video_label)
        
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet(
            "color: #0f0; font-weight: bold; font-size: 14px;"
        )
        left_layout.addWidget(self.status_bar)
        
        det_layout.addWidget(left_panel, 3)
        
        # Right panel: Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Source selection
        grp_src = QGroupBox("📁 Source")
        lay_src = QVBoxLayout(grp_src)
        self.combo_src = QComboBox()
        self.combo_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.combo_src.currentIndexChanged.connect(self.on_source_change)
        lay_src.addWidget(self.combo_src)
        
        self.btn_select_file = QPushButton("📂 Select File...")
        self.btn_select_file.hide()
        self.btn_select_file.clicked.connect(self.select_file)
        lay_src.addWidget(self.btn_select_file)
        right_layout.addWidget(grp_src)
        
        # Control buttons
        grp_ctrl = QGroupBox("🎮 Control")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ START")
        self.btn_start.setStyleSheet(
            "background: #2ecc71; color: white; font-weight: bold; height: 40px;"
        )
        self.btn_start.clicked.connect(self.start)
        btn_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("⏹ STOP")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "background: #e74c3c; color: white; font-weight: bold; height: 40px;"
        )
        self.btn_stop.clicked.connect(self.stop)
        btn_layout.addWidget(self.btn_stop)
        lay_ctrl.addLayout(btn_layout)
        right_layout.addWidget(grp_ctrl)
        
        # Export options
        grp_export = QGroupBox("💾 Export")
        lay_exp = QHBoxLayout(grp_export)
        
        self.btn_rec = QPushButton("🔴 REC")
        self.btn_rec.setStyleSheet("background: #333; color: white;")
        self.btn_rec.clicked.connect(self.toggle_rec)
        lay_exp.addWidget(self.btn_rec)
        
        self.btn_screenshot = QPushButton("📸 SC")
        self.btn_screenshot.setStyleSheet("background: #333; color: white;")
        self.btn_screenshot.setToolTip("Take screenshot of current frame")
        self.btn_screenshot.clicked.connect(self.take_screenshot)
        lay_exp.addWidget(self.btn_screenshot)
        
        self.btn_json = QPushButton("📝 JSON")
        self.btn_json.setStyleSheet("background: #333; color: white;")
        self.btn_json.clicked.connect(self.save_json)
        lay_exp.addWidget(self.btn_json)
        
        right_layout.addWidget(grp_export)
        
        # Detected objects list
        grp_objects = QGroupBox("🎯 Detected Objects")
        lay_obj = QVBoxLayout(grp_objects)
        self.txt_objects = QTextEdit()
        self.txt_objects.setReadOnly(True)
        self.txt_objects.setStyleSheet(
            "background: #222; color: #0f0; font-family: Consolas; font-size: 12px;"
        )
        self.txt_objects.setPlaceholderText("Objects will appear here...")
        lay_obj.addWidget(self.txt_objects)
        right_layout.addWidget(grp_objects)
        
        right_layout.addStretch()
        det_layout.addWidget(right_panel, 1)
        
        tabs.addTab(tab_detection, "🎥 Detection")
        
        # === TAB 2: SETTINGS ===
        tab_settings = QWidget()
        set_layout = QVBoxLayout(tab_settings)
        
        # Model and Device
        grp_model = QGroupBox("🤖 Model & Device")
        lay_mod = QVBoxLayout(grp_model)
        
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "yolo26n.pt (Nano)",  # Изменено на реальные имена для скачивания, если их нет
            "yolo26s.pt (Small)",
            "yolo26m.pt (Medium)",
            "yolo26l.pt (Large)"
        ])
        self.combo_model.setCurrentIndex(1)
        lay_mod.addWidget(self.combo_model)
        
        self.combo_device = QComboBox()
        self.combo_device.addItems(["CPU", "CUDA"])
        if torch.cuda.is_available():
            self.combo_device.setCurrentIndex(1)
        lay_mod.addWidget(self.combo_device)
        set_layout.addWidget(grp_model)
        
        # Thresholds
        grp_conf = QGroupBox("⚙️ Thresholds")
        lay_conf = QVBoxLayout(grp_conf)
        
        # Confidence slider
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.slider_confidence = QSlider(Qt.Orientation.Horizontal)
        self.slider_confidence.setRange(5, 95)
        self.slider_confidence.setValue(25)
        self.lbl_confidence = QLabel("0.25")
        self.slider_confidence.valueChanged.connect(
            lambda v: self.lbl_confidence.setText(f"{v/100:.2f}")
        )
        conf_layout.addWidget(self.slider_confidence)
        conf_layout.addWidget(self.lbl_confidence)
        lay_conf.addLayout(conf_layout)
        
        # IoU slider
        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU (NMS):"))
        self.slider_iou = QSlider(Qt.Orientation.Horizontal)
        self.slider_iou.setRange(5, 95)
        self.slider_iou.setValue(45)
        self.lbl_iou = QLabel("0.45")
        self.slider_iou.valueChanged.connect(
            lambda v: self.lbl_iou.setText(f"{v/100:.2f}")
        )
        iou_layout.addWidget(self.slider_iou)
        iou_layout.addWidget(self.lbl_iou)
        lay_conf.addLayout(iou_layout)
        
        set_layout.addWidget(grp_conf)
        set_layout.addStretch()
        tabs.addTab(tab_settings, "⚙️ Settings")
        
        main_layout.addWidget(tabs)

    def on_source_change(self, idx: int) -> None:
        self.btn_select_file.setVisible(idx > 0)

    def select_file(self) -> None:
        idx = self.combo_src.currentIndex()
        if idx == 0:
            return
        
        title = ["", "Video", "Image"][idx]
        f_str = ["", "Videos (*.mp4 *.avi *.mkv *.mov *.wmv)", "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"][idx]
        
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {title}", "", f_str)
        if file_path:
            self.source_path = file_path
            self.status_bar.setText(f"File: {pathlib.Path(file_path).name}")

    def start(self) -> None:
        try:
            src_idx = self.combo_src.currentIndex()
            if src_idx > 0 and not self.source_path:
                QMessageBox.warning(self, "Error", "Please select a file!")
                return

            model_name = self.combo_model.currentText().split()[0]
            src_type = ["webcam", "video", "image"][src_idx]
            
            conf = self.slider_confidence.value() / 100.0
            iou = self.slider_iou.value() / 100.0
            device = self.combo_device.currentText().lower()

            self.detection_history = []
            self.fps_history = []
            self.txt_objects.clear()

            self.engine = DetectionEngine(
                source_type=src_type,
                source_path=self.source_path,
                model_name=model_name,
                conf=conf,
                iou=iou,
                imgsz=640,
                device=device
            )
            
            self.engine.frame_ready.connect(self.update_frame)
            self.engine.stats_ready.connect(self.update_stats)
            self.engine.data_ready.connect(self.update_data)
            self.engine.error_occurred.connect(self.handle_error)
            
            # CRITICAL FIX: Подписка на завершение потока, чтобы сбросить UI
            self.engine.finished.connect(self._on_engine_finished)
            
            self.engine.start()
            
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.status_bar.setText("⚡ Processing...")
            
        except Exception as e:
            logger.error(f"❌ Critical error: {e}")
            QMessageBox.critical(self, "Critical Error", str(e))

    def stop(self) -> None:
        if self.engine:
            self.engine.stop()
            self.engine = None
        self._on_engine_finished()
        self.status_bar.setText("⏹️ Stopped")

    def _on_engine_finished(self) -> None:
        """Called safely when the QThread finishes natural execution or stopped"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        # Если велась запись, остановить её и обновить UI
        if self.btn_rec.text() == "⏹ STOP":
            self.toggle_rec()
            
        if self.status_bar.text() == "⚡ Processing...":
            self.status_bar.setText("✅ Finished processing")

    def handle_error(self, error_msg: str) -> None:
        QMessageBox.critical(self, "Error", error_msg)
        self.stop()

    def toggle_rec(self) -> None:
        if self.engine:
            is_stopped = self.engine.toggle_recording()
            if is_stopped:
                self.btn_rec.setText("🔴 REC")
                self.btn_rec.setStyleSheet("background: #333; color: white;")
                QMessageBox.information(self, "Ready", "Video saved to /output")
            else:
                self.btn_rec.setText("⏹ STOP")
                self.btn_rec.setStyleSheet("background: #e74c3c; color: white;")

    def take_screenshot(self) -> None:
        pix = self.video_label.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"shot_{datetime.now().strftime('%H%M%S')}.jpg"
            if pix.save(str(path)):
                self.status_bar.setText(f"📸 Screenshot saved: {path.name}")
                logger.info(f"📸 Screenshot: {path}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save screenshot")

    def save_json(self) -> None:
        if not self.detection_history:
            QMessageBox.warning(self, "No Data", "No data to export")
            return
        try:
            path = OUTPUT_DIR / f"report_{datetime.now().strftime('%H%M%S')}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.detection_history, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", f"Report saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def update_frame(self, q_img) -> None:
        pix = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def update_stats(self, fps: float, lat: float, count: int) -> None:
        self.fps_history.append(fps)
        if len(self.fps_history) > 100:
            self.fps_history.pop(0)
        self.status_bar.setText(f"FPS: {fps:.1f} | Latency: {lat:.1f}ms | Objects: {count}")

    def update_data(self, frame_data: list) -> None:
        if len(self.detection_history) < 10000:
            self.detection_history.append(frame_data)
        
        if frame_data:
            counts = {}
            for item in frame_data:
                name = item.get('class_name', 'unknown')
                counts[name] = counts.get(name, 0) + 1
            
            text_lines = [f"• {k}: {v}" for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)]
            self.txt_objects.setText("\n".join(text_lines) if text_lines else "No objects")
        else:
            self.txt_objects.clear()

    def closeEvent(self, event) -> None:
        self.stop()
        logger.info("👋 Application closed")
        event.accept()