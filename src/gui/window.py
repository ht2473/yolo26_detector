"""Main application window - Detection Module Only"""
import sys
import pathlib
import json
from datetime import datetime
from typing import Optional, List, Dict
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QGroupBox, QStatusBar,
    QFileDialog, QTextEdit, QFormLayout, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage
from src.core.engine import DetectionEngine
from src.utils.helpers import ensure_dir
from loguru import logger

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO26 Detector PRO")
        self.setMinimumSize(1100, 750)
        ensure_dir(MODELS_DIR)
        ensure_dir(OUTPUT_DIR)

        self.detection_engine: Optional[DetectionEngine] = None
        self.source_path: str = ""  # 🔑 FIX: Инициализация атрибута
        self.active_module: bool = False
        self.detection_history: List[Dict] = []

        self.det_config = {
            "source_type": "webcam",
            "source_path": "",
            "model_name": "yolo26s.pt",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "conf": 0.25,
            "iou": 0.45,
            "imgsz": 640
        }

        self._init_ui()
        logger.info("🚀 Application started (Detection Module)")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        main_layout.addWidget(self._create_detection_tab())

        self.global_status = QStatusBar()
        self.global_status.setStyleSheet("color: #0f0; background: #111; font-weight: 500;")
        self.global_status.showMessage("✅ Ready — Select source and click START")
        self.setStatusBar(self.global_status)

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

        # Right: Controls
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(12)

        # Source
        grp_src = self._group_box("📁 Source")
        lay_src = QVBoxLayout(grp_src)
        self.det_combo_src = QComboBox()
        self.det_combo_src.addItems(["📹 Webcam", "🎬 Video File", "🖼️ Image"])
        self.det_combo_src.currentIndexChanged.connect(self._on_source_change)
        lay_src.addWidget(self.det_combo_src)
        self.det_select_file = QPushButton("📂 Select File...")
        self.det_select_file.hide()
        self.det_select_file.clicked.connect(self._select_file)
        lay_src.addWidget(self.det_select_file)
        right_layout.addWidget(grp_src)

        # Model & Device
        grp_model = self._group_box("🤖 Model & Device")
        lay_mod = QFormLayout(grp_model)
        self.det_combo_model = QComboBox()
        self.det_combo_model.addItems(["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt"])
        self.det_combo_model.setCurrentText("yolo26s.pt")
        lay_mod.addRow("Model:", self.det_combo_model)

        self.det_combo_device = QComboBox()
        self.det_combo_device.addItems(["CPU", "CUDA"])
        if torch.cuda.is_available():
            self.det_combo_device.setCurrentIndex(1)
        lay_mod.addRow("Device:", self.det_combo_device)
        right_layout.addWidget(grp_model)

        # Thresholds
        grp_thr = self._group_box("⚙️ Thresholds")
        lay_thr = QFormLayout(grp_thr)
        self.det_slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.det_slider_conf.setRange(5, 95)
        self.det_slider_conf.setValue(25)
        self.det_lbl_conf = QLabel("0.25")
        self.det_slider_conf.valueChanged.connect(lambda v: self.det_lbl_conf.setText(f"{v/100:.2f}"))
        lay_thr.addRow("Confidence:", self._slider_with_label(self.det_slider_conf, self.det_lbl_conf))

        self.det_slider_iou = QSlider(Qt.Orientation.Horizontal)
        self.det_slider_iou.setRange(5, 95)
        self.det_slider_iou.setValue(45)
        self.det_lbl_iou = QLabel("0.45")
        self.det_slider_iou.valueChanged.connect(lambda v: self.det_lbl_iou.setText(f"{v/100:.2f}"))
        lay_thr.addRow("IoU (NMS):", self._slider_with_label(self.det_slider_iou, self.det_lbl_iou))

        self.det_spin_imgsz = QLineEdit("640")
        lay_thr.addRow("Input Size:", self.det_spin_imgsz)
        right_layout.addWidget(grp_thr)

        # Control
        grp_ctrl = self._group_box("🎮 Control")
        lay_ctrl = QVBoxLayout(grp_ctrl)
        btn_row = QHBoxLayout()
        self.det_start = QPushButton("▶ START")
        self.det_start.setStyleSheet(self._btn_style("#2ecc71"))
        self.det_start.clicked.connect(self._start_detection)
        btn_row.addWidget(self.det_start)

        self.det_stop = QPushButton("⏹ STOP")
        self.det_stop.setEnabled(False)
        self.det_stop.setStyleSheet(self._btn_style("#e74c3c"))
        self.det_stop.clicked.connect(self._stop_detection)
        btn_row.addWidget(self.det_stop)
        lay_ctrl.addLayout(btn_row)
        right_layout.addWidget(grp_ctrl)

        # Export
        grp_exp = self._group_box("💾 Export")
        lay_exp = QHBoxLayout(grp_exp)
        self.det_rec = QPushButton("🔴 REC")
        self.det_rec.setStyleSheet(self._btn_style("#333"))
        self.det_rec.clicked.connect(self._toggle_rec)
        lay_exp.addWidget(self.det_rec)

        self.det_shot = QPushButton("📸 SC")
        self.det_shot.setStyleSheet(self._btn_style("#333"))
        self.det_shot.clicked.connect(self._take_screenshot)
        lay_exp.addWidget(self.det_shot)

        self.det_json = QPushButton("📝 JSON")
        self.det_json.setStyleSheet(self._btn_style("#333"))
        self.det_json.clicked.connect(self._save_json)
        lay_exp.addWidget(self.det_json)
        right_layout.addWidget(grp_exp)

        # Results
        grp_res = self._group_box("🎯 Detected Objects")
        lay_res = QVBoxLayout(grp_res)
        self.det_results = QTextEdit()
        self.det_results.setReadOnly(True)
        self.det_results.setStyleSheet("background: #111; color: #0f0; font-family: Consolas; font-size: 11px;")
        lay_res.addWidget(self.det_results)
        right_layout.addWidget(grp_res)

        right_layout.addStretch()
        layout.addWidget(right, 1)
        return tab

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
        # 🔑 FIX: Убрана невалидная конструкция {bg}cc, заменена на opacity
        return f"""
            QPushButton {{
                background: {bg}; color: white; border: none;
                border-radius: 6px; padding: 8px 16px; font-weight: 500;
            }}
            QPushButton:hover {{ background: {bg}; opacity: 0.9; border: 1px solid rgba(255,255,255,0.2); }}
            QPushButton:disabled {{ background: #444; color: #888; }}
        """

    def _slider_with_label(self, slider: QSlider, label: QLabel) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(slider)
        lay.addWidget(label)
        return w

    def _on_source_change(self, idx: int):
        self.det_select_file.setVisible(idx > 0)
        if idx == 0:
            self.source_path = ""
            self.det_status.setText("📹 Webcam selected")

    def _select_file(self):
        idx = self.det_combo_src.currentIndex()
        if idx == 0: return
        title = ["", "Video", "Image"][idx]
        ext = ["", "Videos (*.mp4 *.avi *.mkv)", "Images (*.jpg *.png)"][idx]
        file, _ = QFileDialog.getOpenFileName(self, f"Select {title}", "", ext)
        if file:
            self.source_path = file
            self.det_status.setText(f"📄 {pathlib.Path(file).name}")

    def _start_detection(self):
        if self.active_module:
            self._stop_detection()

        try:
            source_type = ["webcam", "video", "image"][self.det_combo_src.currentIndex()]
            self.det_config.update({
                "source_type": source_type,
                "source_path": self.source_path if source_type in ("video", "image") else "",
                "model_name": self.det_combo_model.currentText(),
                "device": self.det_combo_device.currentText().lower(),
                "conf": self.det_slider_conf.value() / 100.0,
                "iou": self.det_slider_iou.value() / 100.0,
                "imgsz": int(self.det_spin_imgsz.text() or 640)
            })

            self.detection_engine = DetectionEngine(**self.det_config)
            self.detection_history.clear()
            self.det_results.clear()

            self.detection_engine.frame_ready.connect(self._update_frame)
            self.detection_engine.stats_ready.connect(self._update_stats)
            self.detection_engine.data_ready.connect(self._update_results)
            self.detection_engine.error_occurred.connect(self._handle_error)
            self.detection_engine.finished.connect(self._on_engine_finished)

            self.detection_engine.start()
            self.active_module = True

            self.det_start.setEnabled(False)
            self.det_stop.setEnabled(True)
            self.det_status.setText("⚡ Processing...")
            self.global_status.showMessage("🔥 Detection module running")

        except Exception as e:
            logger.error(f"❌ Start failed: {e}")
            QMessageBox.critical(self, "Error", str(e))
            self._stop_detection()

    def _stop_detection(self):
        if self.detection_engine and self.detection_engine.isRunning():
            self.detection_engine.stop()

        self.det_start.setEnabled(True)
        self.det_stop.setEnabled(False)
        self.det_status.setText("⏹️ Stopped")
        if self.det_rec.text() == "⏹ STOP":
            self.det_rec.setText("🔴 REC")
            self.det_rec.setStyleSheet(self._btn_style("#333"))

        self.active_module = False
        self.global_status.showMessage("✅ Module stopped — Ready")

    def _on_engine_finished(self):
        self.det_start.setEnabled(True)
        self.det_stop.setEnabled(False)
        if self.det_status.text() == "⚡ Processing...":
            self.det_status.setText("✅ Finished")
        self.active_module = False

    def _handle_error(self, error_msg: str):
        QMessageBox.critical(self, "Detection Error", error_msg)
        self._stop_detection()

    def _toggle_rec(self):
        if self.detection_engine and self.detection_engine.isRunning():
            is_stopped = self.detection_engine.toggle_recording()
            if is_stopped:
                self.det_rec.setText("🔴 REC")
                self.det_rec.setStyleSheet(self._btn_style("#333"))
                QMessageBox.information(self, "Ready", f"Video saved to {OUTPUT_DIR}")
            else:
                self.det_rec.setText("⏹ STOP")
                self.det_rec.setStyleSheet(self._btn_style("#e74c3c"))

    def _take_screenshot(self):
        pix = self.det_video.pixmap()
        if pix and not pix.isNull():
            path = OUTPUT_DIR / f"det_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            if pix.save(str(path)):
                self.global_status.showMessage(f"📸 Saved: {path.name}")
            else:
                QMessageBox.critical(self, "Error", "Failed to save screenshot")

    def _save_json(self):
        if not self.detection_history:
            QMessageBox.warning(self, "No Data", "No detections to export")
            return
        try:
            path = OUTPUT_DIR / f"detection_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.detection_history, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Success", f"Report saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def _update_frame(self, q_img: QImage):
        pix = QPixmap.fromImage(q_img)
        self.det_video.setPixmap(pix.scaled(
            self.det_video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def _update_stats(self, fps: float, latency: float, count: int):
        self.det_status.setText(f"FPS: {fps:.1f} | Lat: {latency:.1f}ms | Obj: {count}")

    def _update_results(self, data: List[Dict]):
        if not data:
            return
            
        self.detection_history.extend(data[:1000])

        lines = []
        for item in data[:10]:
            name = item.get("class_name", "unknown")
            conf = item.get("confidence", 1.0)
            bbox = item.get("bbox", [])
            lines.append(f"• {name} ({conf:.0%}) [{bbox[:2] if bbox else ''}]")
        self.det_results.setText("\n".join(lines))

    def closeEvent(self, event):
        self._stop_detection()
        logger.info("👋 Application closed")
        event.accept()

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
        QPushButton { border: none; border-radius: 5px; padding: 6px 12px; }
        QTextEdit { background: #111; border: 1px solid #333; border-radius: 4px; font-family: Consolas; font-size: 11px; }
        QScrollBar { background: #222; width: 10px; }
        QScrollBar::handle { background: #555; border-radius: 4px; }
    """)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()