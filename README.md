# YOLO26 Detector PRO

Desktop-приложение для детекции объектов в реальном времени на базе нейросети YOLO26.  

---

## 🚀 Возможности
- **Источники**: веб-камера, видеофайл, изображение
- **Модели**: YOLO26 Nano / Small / Medium / Large
- **Настройки**: пороги Confidence и IoU, размер входа, выбор CPU/CUDA
- **Экспорт**: запись видео, скриншоты, JSON-отчёты с координатами
- **Интерфейс**: современный GUI на PyQt6 с live-статистикой (FPS, Latency, объекты)

---

## 📦 Установка

1. **Требования**: Python 3.11+, Windows 10/11, NVIDIA GPU с CUDA (опционально)
2. **Создайте окружение и установите зависимости**:
   ```bash
   git clone https://github.com/ht2473/yolo26_detector.git
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   python main.py 
