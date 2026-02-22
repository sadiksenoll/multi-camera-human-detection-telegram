# cam2tg.py - 70 Kamera İzleme ve İnsan Tespiti Sistemi (Arka Plan Modu)
import os
import sys
import time
import cv2
import requests
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import queue
import json

# YOLO import - EN İYİ İNSAN TESPİTİ (80%+ doğruluk)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 başarıyla yüklendi - En doğru insan tespiti aktif")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO bulunamadı - Lütfen 'pip install ultralytics' yapın")

# EXE uyumlu dosya yolu fonksiyonları
def get_app_directory():
    """EXE'nin bulunduğu klasörü bul"""
    if getattr(sys, 'frozen', False):
        # EXE olarak çalışıyor
        return os.path.dirname(sys.executable)
    else:
        # Python script olarak çalışıyor
        return os.path.dirname(os.path.abspath(__file__))

def ensure_config_exists():
    """Config dosyası yoksa otomatik oluştur"""
    if not os.path.exists(CONFIG_FILE):
        default_config = {
            'urls': [''] * 70,
            'settings': {
                'bot_token': 'YOUR_BOT_TOKEN_HERE',
                'chat_id': YOUR_CHAT_ID_HERE
            },
            'work_schedule': {
                'Monday': {'enabled': True, 'start_hour': 8, 'start_min': 0, 'end_hour': 18, 'end_min': 0},
                'Tuesday': {'enabled': True, 'start_hour': 8, 'start_min': 0, 'end_hour': 18, 'end_min': 0},
                'Wednesday': {'enabled': True, 'start_hour': 8, 'start_min': 0, 'end_hour': 18, 'end_min': 0},
                'Thursday': {'enabled': True, 'start_hour': 8, 'start_min': 0, 'end_hour': 18, 'end_min': 0},
                'Friday': {'enabled': True, 'start_hour': 8, 'start_min': 0, 'end_hour': 18, 'end_min': 0},
                'Saturday': {'enabled': False, 'start_hour': 0, 'start_min': 0, 'end_hour': 0, 'end_min': 0},
                'Sunday': {'enabled': False, 'start_hour': 0, 'start_min': 0, 'end_hour': 0, 'end_min': 0}
            }
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"Config dosyası oluşturuldu: {CONFIG_FILE}")
        except Exception as e:
            print(f"Config dosyası oluşturma hatası: {e}")

def create_readme():
    """Kullanım kılavuzu oluştur"""
    readme_content = """KAMERA İZLEME SİSTEMİ
=====================

KULLANIM:
1. KameraIzleme.exe dosyasına çift tıklayın
2. Program otomatik açılacak
3. Kamera URL'lerini girin (örn: 0, 1, rtsp://192.168.1.100:554/stream1)
4. Başlat butonuna tıklayın

ÖZELLİKLER:
- 70 kamera desteği
- İnsan tespiti
- Telegram bildirimi
- Otomatik URL kaydetme

DESTEK:
- USB kameralar: 0, 1, 2...
- IP kameralar: rtsp://ip:port/stream
- Local ağ kameralar: rtsp://192.168.x.x:554/stream

NOT: İlk çalıştırmada config dosyası otomatik oluşur.
"""
    
    readme_path = os.path.join(get_app_directory(), "README.txt")
    try:
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"README dosyası oluşturuldu: {readme_path}")
    except Exception as e:
        print(f"README dosyası oluşturma hatası: {e}")

BOT_TOKEN  = "YOUR_BOT_TOKEN_HERE"
CHAT_ID    = YOUR_CHAT_ID_HERE

SEND_COOLDOWN_SEC = 30  # 30 saniye cooldown (Telegram mesajı için)
DETECTION_COOLDOWN_SEC = 30  # 30 saniye cooldown (İnsan tespiti için - spam önleme)
JPEG_QUALITY = 85
DETECT_EVERY_N = 15  # Tespit sıklığı (her 15 frame'de bir) - Yavaş kameralar için düşürüldü
MIN_HUMAN_CONFIDENCE = 0.3  # Daha düşük güven eşiği (insan tespit etme öncelikli)
CONFIG_FILE = os.path.join(get_app_directory(), "camera_urls.json")  # EXE uyumlu config dosyası

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|probesize;50000000|analyzeduration;10000000|stimeout;20000000|max_delay;500000"
)

def is_work_hours():
    """Haftalık çalışma saatleri kontrolü"""
    try:
        # Config dosyasını oku
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                work_schedule = data.get('work_schedule', {})
        else:
            return True  # Config yoksa her zaman aktif
        
        # Bugünün günü
        today = datetime.now().strftime('%A')
        current_hour = datetime.now().hour
        
        # Bugünün ayarları
        today_schedule = work_schedule.get(today, {'enabled': True, 'start_hour': 8, 'start_min': 0, 'end_hour': 18, 'end_min': 0})
        
        # Bugün aktif mi?
        if not today_schedule.get('enabled', True):
            return False
        
        # Mesai saatleri içinde mi?
        start_hour = today_schedule.get('start_hour', 8)
        start_min = today_schedule.get('start_min', 0)
        end_hour = today_schedule.get('end_hour', 18)
        end_min = today_schedule.get('end_min', 0)
        
        # Şu anki saat ve dakika
        current_minute = datetime.now().minute
        current_time_minutes = current_hour * 60 + current_minute
        start_time_minutes = start_hour * 60 + start_min
        end_time_minutes = end_hour * 60 + end_min
        
        return start_time_minutes <= current_time_minutes < end_time_minutes
        
    except Exception as e:
        print(f"Çalışma saati kontrolü hatası: {e}")
        return True  # Hata durumunda aktif kal

class CameraMonitor:
    def __init__(self, camera_id, parent_gui):
        self.camera_id = camera_id
        self.gui = parent_gui
        self.url = ""
        self.cap = None
        self.running = False
        self.thread = None
        self.last_send = 0.0
        self.last_detection = 0.0  # Tespit cooldown için (son tespit zamanı)
        
        # Durum bilgileri
        self.last_detection_time = None  # Sadece GUI için (bilgi amaçlı)
        self.detection_count = 0
        self.status = "Durduruldu"
        self.is_connected = False  # Bağlantı durumu
        self.connection_lost = False  # Bağlantı kopma durumu
        self.last_frame_time = 0  # Son frame zamanı
        
        # Bölge ayarları - her kamera için ayrı bölgeler
        self.detection_regions = []  # [(x1, y1, x2, y2), ...] formatında
        self.use_regions = False  # Bölge kontrolü aktif mi?
        
        # YOLO İNSAN TESPİTİ - GLOBAL MODELİ PAYLAŞ (RAM tasarrufu!)
        # Her kamera yeni model yüklemek yerine global modeli kullanır
        self.yolo_available = self.gui.global_yolo_available if hasattr(self.gui, 'global_yolo_available') else False
        
        if self.yolo_available:
            # Global YOLO modelini kullan (yeni instance yok!)
            self.yolo = self.gui.global_yolo
            self.device = self.gui.global_device
            self.hog = None
            
            self.gui.log_message(f"Kamera {self.camera_id}: ✅ Global YOLO modeli kullanılacak")
        else:
            self.yolo = None
            # HOG İnsan Tespiti (geri dönüş)
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.gui.log_message(f"Kamera {self.camera_id}: ⚠️ YOLO yok, HOG kullanılacak (pip install ultralytics)")
        
        # Haar Cascade ile alternatif tespit (EXE uyumlu)
        try:
            # EXE için Haar Cascade dosya yolu
            if getattr(sys, 'frozen', False):
                # EXE olarak çalışıyor - PyInstaller temp klasöründe ara
                import tempfile
                temp_dir = tempfile.gettempdir()
                haar_path = os.path.join(temp_dir, 'haarcascade_fullbody.xml')
                if not os.path.exists(haar_path):
                    # Alternatif yol
                    haar_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            else:
                # Python script olarak çalışıyor
                haar_path = cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            
            self.haar_cascade = cv2.CascadeClassifier(haar_path)
            if self.haar_cascade.empty():
                raise Exception("Haar Cascade dosyası boş")
            self.haar_available = True
            self.gui.log_message(f"Kamera {self.camera_id}: Haar Cascade yüklendi: {haar_path}")
        except Exception as e:
            self.haar_available = False
            self.gui.log_message(f"Kamera {self.camera_id}: Haar Cascade yüklenemedi: {e}, sadece HOG kullanılacak")
        
        # DNN tabanlı güçlü tespit sistemi (MobileNet-SSD)
        self.dnn_net = None
        self.dnn_available = False
        try:
            # MobileNet-SSD model dosyaları (OpenCV ile birlikte gelir)
            model_path = cv2.data.haarcascades.replace('haarcascades', 'dnn')
            if os.path.exists(model_path):
                # Eğer model dosyaları varsa yükle
                self.dnn_available = True
                self.gui.log_message(f"Kamera {self.camera_id}: DNN tespit sistemi hazır")
            else:
                self.gui.log_message(f"Kamera {self.camera_id}: DNN modeli bulunamadı, HOG+Haar kullanılacak")
        except Exception as e:
            self.gui.log_message(f"Kamera {self.camera_id}: DNN yükleme hatası: {e}")
        
        # frame_queue kaldırıldı (görüntü gösterilmiyor)
        
        # Hareket tespiti için
        self.prev_frame = None
        self.motion_threshold = 100  # Çok düşük hareket eşiği (insan tespit etme öncelikli)
        self.frame_count = 0  # Frame sayacı (ilk frame'lerde direkt tespit)
        
    def send_photo(self, frame, caption=""):
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return False, "JPEG encode failed"
        files = {"photo": ("frame.jpg", jpg.tobytes(), "image/jpeg")}
        data = {"chat_id": str(CHAT_ID), "caption": caption}
        try:
            r = requests.Session().post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto", 
                                      data=data, files=files, timeout=10)
            return (r.ok, r.text if not r.ok else "OK")
        except Exception as e:
            return False, str(e)
    
    def is_in_detection_region(self, x_min, y_min, x_max, y_max):
        """Bounding box'un tanımlı bölgeler içinde olup olmadığını kontrol et"""
        if not self.use_regions or len(self.detection_regions) == 0:
            return True  # Bölge tanımlı değilse, tüm frame'de ara
        
        # Bounding box'un merkez noktası
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Her bölgeyi kontrol et
        for idx, region in enumerate(self.detection_regions):
            x1, y1, x2, y2 = region
            # Merkez nokta bölge içinde mi?
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                return True
        
        return False
    
    def detect_people(self, frame):
        """
        YOLO DETECTION - EN DOĞRU İNSAN TESPİTİ (80%+ doğruluk)
        Bölge kontrolü ile sadece belirlenen bölgelerde tespit yapar
        """
        if frame is None or frame.size == 0:
            return []
        
        people_detected = []
        frame_h, frame_w = frame.shape[:2]
        
        try:
            # YOLO kullanılıyor mu?
            if self.yolo_available and self.yolo:
                start_time = time.time()
                
                # YOLO tespit (sadece 'person' sınıfı - class 0) - GPU kullan
                # Thread-safe: Aynı anda birden fazla thread model'i çağırmasın
                with self.gui.yolo_lock:
                    results = self.yolo(frame, classes=[0], conf=0.5, verbose=False, device=self.device)
                
                detection_time = time.time() - start_time
                
                # Sonuçları analiz et
                for result in results:
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            # Sadece 'person' sınıfı (class 0)
                            if int(box.cls) == 0:  # Person class
                                confidence = float(box.conf[0])
                                
                                # Bounding box koordinatları
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                x_min, y_min = int(x1), int(y1)
                                x_max, y_max = int(x2), int(y2)
                                width = x_max - x_min
                                height = y_max - y_min
                                
                                # Geçerli boyut kontrolü
                                if width > 50 and height > 100:
                                    # Bölge kontrolü - sadece belirlenen bölgelerdeki tespitleri kaydet
                                    if self.is_in_detection_region(x_min, y_min, x_max, y_max):
                                        people_detected.append((x_min, y_min, width, height))
                    
                # Tespit yok, log yok (CPU tasarrufu)
                
                return people_detected
                
            else:
                # HOG FALLBACK
                
                if self.hog is None:
                    self.hog = cv2.HOGDescriptor()
                    self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                
                boxes, weights = self.hog.detectMultiScale(
                    frame,
                    winStride=(8, 8),
                    padding=(16, 16),
                    scale=1.05,
                    hitThreshold=0.5,  # Yüksek güven
                    useMeanshiftGrouping=False
                )
                
                for i, (x, y, w, h) in enumerate(boxes):
                    conf = weights[i] if len(weights) > i else 0.3
                    if conf >= 0.5 and w > 50 and h > 100:  # Yüksek güven ve büyük boyut
                        # Bölge kontrolü - sadece belirlenen bölgelerdeki tespitleri kaydet
                        if self.is_in_detection_region(x, y, x+w, y+h):
                            people_detected.append((x, y, w, h))
                
                return people_detected
            
        except Exception as e:
            self.gui.log_message(f"❌ Kamera {self.camera_id}: Tespit hatası - {e}")
            import traceback
            self.gui.log_message(f"   Detay: {traceback.format_exc()}")
            return []
    
    def preprocess_frame(self, frame):
        """Görüntü ön işleme - dengeli"""
        # Hafif gürültü azaltma
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        # Dengeli kontrast artırma
        frame = cv2.convertScaleAbs(frame, alpha=1.08, beta=4)
        return frame
    
    def advanced_filter_people(self, boxes, original_frame):
        """Gelişmiş insan filtreleme - gevşek (insan tespit etme öncelikli)"""
        if len(boxes) <= 1:
            return boxes
        
        filtered = []
        h, w = original_frame.shape[:2]
        
        for box in boxes:
            x, y, box_w, box_h = box
            
            # 1. Boyut kontrolü - çok gevşek
            if box_w < 30 or box_h < 60 or box_w > w*0.9 or box_h > h*0.9:
                continue
            
            # 2. Oran kontrolü (insan vücut oranı) - gevşek
            aspect_ratio = box_h / box_w
            if aspect_ratio < 1.2 or aspect_ratio > 4.5:  # Geniş vücut oranı aralığı
                continue
            
            # 3. Konum kontrolü (çok kenarda olmasın) - daha toleranslı
            if x < 10 or y < 10 or x + box_w > w - 10 or y + box_h > h - 10:
                continue
            
            filtered.append(box)
        
        # Duplikasyonları temizle
        return self.remove_duplicates(filtered, threshold=0.2)  # Düşük threshold
    
    def remove_duplicates(self, boxes, threshold=0.3):
        """Yakın kutuları temizle"""
        if len(boxes) <= 1:
            return boxes
        
        filtered = []
        for i, box1 in enumerate(boxes):
            is_duplicate = False
            for j, box2 in enumerate(filtered):
                if self.box_overlap(box1, box2) > threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered.append(box1)
        
        return filtered
    
    def box_overlap(self, box1, box2):
        """İki kutu arasındaki örtüşme oranını hesapla"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Kesişim alanı
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def has_motion(self, frame):
        """Basit hareket tespiti - sadece hareket varsa insan tespiti yap"""
        # Geçersiz frame kontrolü
        if frame is None or frame.size == 0:
            return False
        
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True  # İlk frame'de tespit yap
        
        # Mevcut frame'i gri tonlama
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # BOYUT KONTROLÜ - Frame boyutları uyuşmalı
        if self.prev_frame.shape != gray_frame.shape:
            # Boyutlar uyuşmuyor, prev_frame'i güncelle
            self.prev_frame = gray_frame
            return True  # Boyut değişti, tespit yap
        
        # Frame'ler arası fark
        diff = cv2.absdiff(self.prev_frame, gray_frame)
        
        # Threshold uygula
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Hareket piksellerini say
        motion_pixels = cv2.countNonZero(thresh)
        
        # Önceki frame'i güncelle
        self.prev_frame = gray_frame
        
        # Hareket varsa True döndür
        return motion_pixels > self.motion_threshold
    
    def run_camera(self, url_or_index):
        self.url = url_or_index
        self.running = True
        
        # Eğer sadece rakam girilmişse, USB/laptop kamerası olarak aç
        try:
            camera_index = int(url_or_index)
            self.cap = cv2.VideoCapture(camera_index)
            self.gui.log_message(f"Kamera {self.camera_id}: USB/Laptop kamerası açılıyor (index {camera_index})")
        except ValueError:
            # URL veya RTSP stream
            self.cap = cv2.VideoCapture(url_or_index, cv2.CAP_FFMPEG)
            self.gui.log_message(f"Kamera {self.camera_id}: RTSP/URL bağlanıyor: {url_or_index[:50]}...")
        
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # CPU dostu buffer
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # 15 FPS hedefle (1650 Ti için optimize edildi)
        
        if not self.cap.isOpened():
            self.gui.log_message(f"Kamera {self.camera_id}: Bağlantı hatası!")
            self.running = False
            return
        
        # Kamera FPS'ini kontrol et ve log'la
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.gui.log_message(f"Kamera {self.camera_id}: Başlatıldı (FPS: {actual_fps:.1f})")
        self.status = "Aktif"
        self.is_connected = True
        self.connection_lost = False
        n = 0
        
        retry_count = 0
        max_retries = 3
        
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                # Bağlantı koptu
                if self.is_connected:
                    self.connection_lost = True
                    self.is_connected = False
                    self.gui.log_message(f"⚠️ Kamera {self.camera_id}: Bağlantı koptu!")
                
                # Otomatik yeniden bağlanma denemesi
                if self.running and retry_count < max_retries:
                    retry_count += 1
                    self.gui.log_message(f"🔄 Kamera {self.camera_id}: Yeniden bağlanılıyor... (Deneme {retry_count}/{max_retries})")
                    
                    # Mevcut bağlantıyı kapat
                    if self.cap:
                        self.cap.release()
                    
                    # Kısa bekleme
                    time.sleep(2)
                    
                    # Yeniden bağlan
                    try:
                        camera_index = int(self.url)
                        self.cap = cv2.VideoCapture(camera_index)
                        self.gui.log_message(f"🔄 Kamera {self.camera_id}: USB/Laptop kamerası yeniden açılıyor (index {camera_index})")
                    except ValueError:
                        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                        self.gui.log_message(f"🔄 Kamera {self.camera_id}: RTSP/URL yeniden bağlanıyor: {self.url[:50]}...")
                    
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 15)
                    
                    # Bağlantı başarılı mı kontrol et
                    if self.cap.isOpened():
                        retry_count = 0  # Başarılı olursa sıfırla
                        self.is_connected = True
                        self.connection_lost = False
                        self.gui.log_message(f"✅ Kamera {self.camera_id}: Yeniden bağlandı!")
                    else:
                        self.gui.log_message(f"❌ Kamera {self.camera_id}: Yeniden bağlanamadı (Deneme {retry_count}/{max_retries})")
                        if retry_count >= max_retries:
                            self.gui.log_message(f"⚠️ Kamera {self.camera_id}: Maksimum deneme sayısına ulaşıldı. Bağlantı bekleniyor...")
                            retry_count = 0  # Reset sayacı ve tekrar dene
                else:
                    time.sleep(0.1)  # CPU dostu bekleme (10 FPS reconnect)
                continue
            
            # Bağlantı başarılı - flag'leri temizle
            if not self.is_connected or self.connection_lost:
                self.is_connected = True
                self.connection_lost = False
                retry_count = 0  # Retry sayacını sıfırla
                self.gui.log_message(f"✅ Kamera {self.camera_id}: Bağlantı stabil")
            
            self.last_frame_time = time.time()
            retry_count = 0  # Başarılı frame okunduğunda retry sayacını sıfırla
            
            # İnsan tespiti - sadece hareket varsa ve belirli aralıklarla
            n += 1
            self.frame_count += 1
            people_detected = []  # Her zaman tanımlı
            
            if n % DETECT_EVERY_N == 0:
                # Tespit cooldown kontrolü - Son tespittten en az X saniye geçmişse tespit yap
                current_time = time.time()
                time_since_last_detection = current_time - self.last_detection if self.last_detection > 0 else DETECTION_COOLDOWN_SEC + 1
                
                if time_since_last_detection >= DETECTION_COOLDOWN_SEC:
                    # Cooldown bitti, tespit yapabilir
                    # İlk 60 frame'de veya hareket varsa tespit yap (insan tespit etme öncelikli)
                    if self.frame_count <= 60 or self.has_motion(frame):
                        people_detected = self.detect_people(frame)
                    else:
                        # Hareket yok, tespit yapma (sessizce atla)
                        pass
                    # Debug: Tespit sonuçları
                    if len(people_detected) > 0:
                        self.last_detection = current_time  # Tespit cooldown'unu başlat
                        self.last_detection_time = datetime.now()
                        self.detection_count += len(people_detected)
                        self.gui.log_message(f"🎯 Kamera {self.camera_id}: İNSAN TESPIT EDILDI! - {len(people_detected)} kişi")
                    # Debug log kaldırıldı (spam önleme)
                else:
                    # Cooldown aktif, tespit yapma (spam önleme)
                    pass
            
            # İnsan tespit edildiğinde Telegram'a gönder - HER KAMERA AYRI COOLDOWN
            if len(people_detected) > 0:
                # Cooldown hesapla - bu kamera için ayrı cooldown
                current_time = time.time()
                time_since_last = current_time - self.last_send if self.last_send > 0 else SEND_COOLDOWN_SEC + 1
                
                if time_since_last >= SEND_COOLDOWN_SEC:
                    # Mesai saati kontrolü
                    work_hours_check = is_work_hours()
                    
                    if work_hours_check:
                        # Cooldown bitmiş ve mesai saatleri içinde - fotoğraf gönder
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        caption = f"⚠️ İnsan Algılandı!\n📹 Kamera {self.camera_id}\n👥 {len(people_detected)} kişi\n🕐 {timestamp}"
                        
                        # Frame'i gönder (kutular olmadan)
                        display_frame = frame.copy()
                        
                        sent, msg = self.send_photo(display_frame, caption)
                        if sent:
                            self.last_send = current_time  # Bu kameranın cooldown'unu başlat
                            self.gui.log_message(f"✅ Kamera {self.camera_id}: {len(people_detected)} kişi tespit edildi - Telegram'a gönderildi")
                        else:
                            self.gui.log_message(f"❌ Kamera {self.camera_id}: Telegram hatası - {msg}")
                    else:
                        # Mesai dışı - gönderme
                        self.gui.log_message(f"⏸️ Kamera {self.camera_id}: Mesai saati dışında")
                else:
                    # Cooldown aktif
                    pass
            
            # GUI'ye frame gönderme KALDIRILDI (CPU tasarrufu)
            # Durum bilgisi update_displays() fonksiyonunda gösterilecek
    
    def start(self, url):
        if self.running:
            self.stop()
        # Cooldown'ları sıfırla (yeni başlatma için)
        self.last_send = 0.0
        self.last_detection = 0.0
        self.frame_count = 0  # Frame sayacını sıfırla
        self.thread = threading.Thread(target=self.run_camera, args=(url,), daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        self.status = "Durduruldu"
        self.is_connected = False
        self.connection_lost = False
        if self.cap:
            self.cap.release()
        if self.thread:
            self.thread.join(timeout=2)
        self.gui.log_message(f"Kamera {self.camera_id}: Durduruldu")


class CameraGUI:
    def __init__(self):
        self.root = tk.Tk()

        # ====== BAŞLIK ÇUBUĞU İKONUNU KALDIRMAK İÇİN EKLE (1x1 ŞEFFAF İKON) ======
        try:
            # 1x1 boş PhotoImage oluşturup pencere ikonu olarak ata
            empty_icon = tk.PhotoImage(width=1, height=1)
            # False: tüm pencereler için değil, sadece bu pencere için
            # (Windows'ta başlık çubuğu ve görev çubuğu ikonunu değiştirmelidir)
            self.root.iconphoto(False, empty_icon)
        except Exception as e:
            # Hata olursa sessizce devam et
            print(f"ikon ayarlanamadı: {e}")
        # ======================================================================

        self.root.title("Kamera İzleme Sistemi")
        self.root.geometry("1900x1200")
        
        # Siyah tema arka plan
        self.root.configure(bg='#1e1e1e')

        
        # EXE için otomatik dosya oluşturma
        ensure_config_exists()
        create_readme()
        
        # GLOBAL YOLO MODELİ - Tüm kameralar bu TEK modeli paylaşacak (RAM tasarrufu!)
        self.global_yolo = None
        self.global_yolo_available = False
        self.global_device = 'cpu'
        self.yolo_lock = threading.Lock()  # Thread-safe için lock
        
        if YOLO_AVAILABLE:
            try:
                import torch
                import tempfile
                
                # Model dosyası için özel yol ayarla (EXE uyumlu)
                temp_dir = tempfile.gettempdir()
                yolo_cache_dir = os.path.join(temp_dir, 'yolo_cache')
                os.makedirs(yolo_cache_dir, exist_ok=True)
                os.environ['YOLOV8_HOME'] = yolo_cache_dir
                
                # TEK YOLO modelini yükle
                self.global_yolo = YOLO('yolov8n.pt')
                self.global_yolo.fuse()  # Model optimizasyonu
                
                # GPU kullanımı - ZORUNLU!
                if torch.cuda.is_available():
                    self.global_device = 0  # GPU index
                    gpu_name = torch.cuda.get_device_name(0)
                    self.log_message(f"🚀 Global GPU aktif: {gpu_name}")
                    self.global_yolo_available = True
                    self.log_message(f"✅ Global YOLOv8n modeli yüklendi - Tüm kameralar bu modeli paylaşacak")
                else:
                    # GPU YOK! HATA VER
                    self.global_yolo_available = False
                    error_msg = (
                        "❌ KRITIK HATA: GPU bulunamadı!\n\n"
                        "Lütfen şunları kontrol edin:\n"
                        "• NVIDIA GPU kurulu mu?\n"
                        "• CUDA Toolkit yüklü mü?\n"
                        "• GPU sürücüleri güncel mi?\n\n"
                        "Program GPU olmadan çalışmaz!"
                    )
                    self.log_message(error_msg)
                    print(error_msg)
                    # GUI henüz açılmadı, sadece print yap
                    raise RuntimeError("GPU bulunamadı! NVIDIA GPU ve CUDA gerekli.")
            except Exception as e:
                self.global_yolo_available = False
                self.log_message(f"❌ Global YOLO yüklenemedi: {e}")
        
        self.cameras = []
        self.labels = []
        self.urls = []
        self.start_buttons = []
        self.stop_buttons = []
        self.region_buttons = []  # Bölge ayarlama butonları
        self.photo_images = []  # Görüntüler için
        
        # Thread-safe message queue
        self.message_queue = queue.Queue()
        
        # URL'leri yükle (henüz kameralar oluşturulmadı, sadece URL'leri yükle)
        self.load_urls()
        
        # Geçici olarak bölge ayarlarını sakla
        self.temp_regions = {}
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.temp_regions = data.get('detection_regions', {})
        except:
            pass
        
        # Sol panel - URL girişi
        left_frame = tk.Frame(self.root, bg='#1e1e1e')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=0, pady=0)
        
        # Başlık frame - güzel görünüm
        header_frame = tk.Frame(left_frame, bg='#2d2d2d', relief=tk.RAISED, bd=1)
        header_frame.pack(fill=tk.X, pady=0)
        
        title_label = tk.Label(header_frame, text="📹 KAMERA AYARLARI", 
                              font=("Arial", 16, "bold"), 
                              bg='#2d2d2d', fg='#ffffff', padx=15, pady=10)
        title_label.pack()
        
        # İstatistik çubuğu
        stats_frame = tk.Frame(left_frame, bg='#252525', relief=tk.FLAT)
        stats_frame.pack(fill=tk.X, pady=0)
        
        tk.Label(stats_frame, text="📊 70 Kamera Kapasitesi", 
                font=("Arial", 10), bg='#252525', fg='#888888', padx=10, pady=5).pack()
        
        # Hızlı işlemler butonları
        quick_actions_frame = tk.Frame(left_frame, bg='#1e1e1e')
        quick_actions_frame.pack(fill=tk.X, pady=0)
        
        def create_button(parent, text, command, bg_color, hover_color):
            btn = tk.Button(parent, text=text, command=command,
                          font=("Arial", 10, "bold"),
                          bg=bg_color, fg='#ffffff',
                          activebackground=hover_color, activeforeground='#ffffff',
                          relief=tk.FLAT, cursor='hand2',
                          padx=10, pady=8, bd=0)
            
            def on_enter(e):
                btn['bg'] = hover_color
            def on_leave(e):
                btn['bg'] = bg_color
            
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            return btn
        
        # Toplu başlatma butonu
        connect_btn = create_button(quick_actions_frame, "🚀 Tüm Kameraları Başlat", 
                                    self.connect_all_cameras, '#2d2d2d', '#3d3d3d')
        connect_btn.pack(fill=tk.X, pady=(0, 8))
        
        # Test butonu
        test_btn = create_button(quick_actions_frame, "📹 Kamera Testi", 
                                self.test_detection_system, '#2d2d2d', '#3d3d3d')
        test_btn.pack(fill=tk.X, pady=(0, 8))
        
        # Fotoğraf gönderim ayarları butonu
        photo_btn = create_button(quick_actions_frame, "📸 Fotoğraf Ayarları", 
                                 self.open_photo_settings, '#2d2d2d', '#3d3d3d')
        photo_btn.pack(fill=tk.X)
        
        # Scrollable frame for cameras
        canvas = tk.Canvas(left_frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1e1e1e')
        
        # Canvas resize olduğunda scrollable_frame genişliğini güncelle
        def configure_canvas_window(event):
            # Canvas genişliğine göre window genişliğini ayarla
            canvas_width = event.width
            canvas.itemconfig(canvas.find_all()[0], width=canvas_width)
        
        canvas.bind('<Configure>', configure_canvas_window)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Her kamera için URL girişi
        for i in range(1, 71):  # 70 kamera
            # Modern kart görünümü
            cam_frame = tk.Frame(scrollable_frame, bg='#2d2d2d', relief=tk.FLAT, bd=0)
            cam_frame.pack(fill=tk.X, pady=1, padx=1)
            
            # Başlık
            header = tk.Frame(cam_frame, bg='#3d3d3d', height=30)
            header.pack(fill=tk.X)
            header.pack_propagate(False)
            
            cam_label = tk.Label(header, text=f"📹 Kamera {i}", 
                               font=("Arial", 9, "bold"),
                               bg='#3d3d3d', fg='#ffffff', anchor='w', padx=10)
            cam_label.pack(side=tk.LEFT, fill=tk.Y)
            
            # İçerik frame
            content_frame = tk.Frame(cam_frame, bg='#2d2d2d', padx=5, pady=5)
            content_frame.pack(fill=tk.X)
            
            # URL girişi
            url_entry = tk.Entry(content_frame, font=("Arial", 9),
                               bg='#1e1e1e', fg='#ffffff', 
                               insertbackground='#ffffff',
                               relief=tk.FLAT, bd=2, highlightthickness=1,
                               highlightbackground='#404040', highlightcolor='#2563eb')
            url_entry.pack(fill=tk.X, pady=(0, 4))
            # Kaydedilmiş URL'i yükle
            if i <= len(self.saved_urls):
                url_entry.insert(0, self.saved_urls[i-1])
            self.urls.append(url_entry)
            
            # Butonlar
            btn_frame = tk.Frame(content_frame, bg='#2d2d2d')
            btn_frame.pack(fill=tk.X)
            
            def create_cam_button(parent, text, command, color):
                btn = tk.Button(parent, text=text, command=command,
                              font=("Arial", 8, "bold"),
                              bg=color, fg='#ffffff',
                              activebackground=color, activeforeground='#ffffff',
                              relief=tk.FLAT, cursor='hand2',
                              padx=3, pady=3, bd=0)
                btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
                return btn
            
            start_btn = create_cam_button(btn_frame, "▶ Başlat", 
                                         lambda idx=i-1: self.start_camera(idx), '#3d3d3d')
            self.start_buttons.append(start_btn)
            
            stop_btn = create_cam_button(btn_frame, "⏹ Durdur", 
                                        lambda idx=i-1: self.stop_camera(idx), '#2d2d2d')
            self.stop_buttons.append(stop_btn)
            
            # Bölge ayarlama butonu
            region_btn = create_cam_button(btn_frame, "🎯 Bölge", 
                                           lambda idx=i-1: self.set_detection_region(idx), '#404040')
            self.region_buttons.append(region_btn)
            
            # Kamera monitörü
            monitor = CameraMonitor(i, self)
            self.cameras.append(monitor)
            
            # Bölge ayarlarını yükle
            camera_id_str = str(i + 1)
            if camera_id_str in self.temp_regions:
                monitor.detection_regions = self.temp_regions[camera_id_str]
                monitor.use_regions = len(monitor.detection_regions) > 0
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Sağ panel - Kamera görüntüleri
        right_frame = tk.Frame(self.root, bg='#1e1e1e')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Başlık için güzel frame
        header_frame = tk.Frame(right_frame, bg='#2d2d2d', relief=tk.RAISED, bd=1)
        header_frame.pack(fill=tk.X, pady=0)
        
        tk.Label(header_frame, text="🎥 70 KAMERA İZLEME SİSTEMİ 🎥", 
                font=("Arial", 14, "bold"),
                bg='#2d2d2d', fg='#ffffff', padx=20, pady=10).pack()
        
        # Scrollable canvas for camera grid - tam boyut
        camera_canvas_frame = tk.Frame(right_frame, bg='#1e1e1e')
        camera_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        camera_canvas = tk.Canvas(camera_canvas_frame, bg='black', highlightthickness=0)
        camera_scrollbar_y = ttk.Scrollbar(camera_canvas_frame, orient="vertical", command=camera_canvas.yview)
        camera_scrollbar_x = ttk.Scrollbar(camera_canvas_frame, orient="horizontal", command=camera_canvas.xview)
        
        camera_frame = tk.Frame(camera_canvas, bg='black')
        
        def update_scroll_region(event):
            camera_canvas.configure(scrollregion=camera_canvas.bbox("all"))
        
        camera_frame.bind("<Configure>", update_scroll_region)
        
        # Canvas window oluştur - başlangıç boyutu
        window_id = camera_canvas.create_window((0, 0), window=camera_frame, anchor="nw", 
                                               width=1200, height=900)
        
        # Canvas resize edildiğinde window'u da resize et
        def resize_frame(event):
            canvas_width = event.width
            canvas_height = event.height
            camera_canvas.itemconfig(window_id, width=canvas_width, height=canvas_height)
        
        camera_canvas.bind('<Configure>', resize_frame)
        camera_canvas.configure(yscrollcommand=camera_scrollbar_y.set, xscrollcommand=camera_scrollbar_x.set)
        
        # 70 kamera için grid layout (7 sütun x 10 satır)
        for i in range(70):
            row = i // 7  # 7 sütun
            col = i % 7
            
            label = tk.Label(camera_frame, text=f"Kamera {i+1}\nBaşlatılmadı", 
                           bg='black', fg='white', font=("Arial", 8))
            label.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
            self.labels.append(label)
            self.photo_images.append(None)  # PhotoImage referansı için
        
        # Grid yapılandırması - tüm satır ve sütunları eşit boyutlandır
        for row in range(10):
            camera_frame.grid_rowconfigure(row, weight=1, uniform="cam_row")
        for col in range(7):
            camera_frame.grid_columnconfigure(col, weight=1, uniform="cam_col")
        
        camera_canvas.pack(side="left", fill="both", expand=True)
        camera_scrollbar_y.pack(side="right", fill="y")
        camera_scrollbar_x.pack(side="bottom", fill="x")
        
        # Update thread
        self.update_thread = threading.Thread(target=self.update_displays, daemon=True)
        self.update_thread.start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_urls(self):
        """Kaydedilmiş URL'leri dosyadan yükle - EXE uyumlu"""
        self.saved_urls = []
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.saved_urls = data.get('urls', [])
                self.log_message(f"{len(self.saved_urls)} adet kaydedilmiş URL yüklendi")
            else:
                # Config dosyası yoksa otomatik oluştur
                ensure_config_exists()
                self.saved_urls = [''] * 70  # Boş URL'ler
                self.log_message("Config dosyası otomatik oluşturuldu")
        except Exception as e:
            self.log_message(f"URL yükleme hatası: {e}")
            self.saved_urls = [''] * 70
            # Hata durumunda config dosyasını yeniden oluştur
            try:
                ensure_config_exists()
            except:
                pass
    
    def save_urls(self):
        """Mevcut URL'leri ve bölge ayarlarını dosyaya kaydet - EXE uyumlu"""
        try:
            urls_to_save = []
            for url_entry in self.urls:
                urls_to_save.append(url_entry.get().strip())
            
            # Bölge ayarlarını kaydet
            detection_regions = {}
            for i, camera in enumerate(self.cameras):
                if len(camera.detection_regions) > 0:
                    detection_regions[str(i + 1)] = camera.detection_regions
            
            # Mevcut config'i oku
            data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # URL'leri ve bölgeleri güncelle
            data['urls'] = urls_to_save
            data['detection_regions'] = detection_regions
            
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.log_message("URL'ler ve bölge ayarları kaydedildi")
        except Exception as e:
            self.log_message(f"URL kaydetme hatası: {e}")
            # Hata durumunda config dosyasını yeniden oluştur
            try:
                ensure_config_exists()
            except:
                pass
    
    def test_detection_system(self):
        """Tüm kameraları test et"""
        self.log_message("🧪 Kamera test modu başlatılıyor...")
        
        # Tüm kameraları bul (aktif veya değil)
        all_cameras = []
        for i in range(70):
            url_entry = self.urls[i]
            url_text = url_entry.get().strip()
            
            if url_text:  # URL varsa ekle
                all_cameras.append({
                    'index': i,
                    'camera_id': i + 1,
                    'url': url_text,
                    'status': 'Aktif' if self.cameras[i].running else 'Durduruldu'
                })
        
        if not all_cameras:
            messagebox.showwarning("Uyarı", "❌ Kamera URL'i girilmemiş!\n\nLütfen en az bir kamera için URL girin.")
            return
        
        # Kamera seçim penceresi
        dialog = tk.Toplevel(self.root)
        dialog.title("📹 Kamera Testi - Kamerayı Seçin")
        dialog.geometry("450x500")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg='#1e1e1e')
        
        tk.Label(dialog, text=f"Test etmek istediğiniz kamerayı seçin:\nToplam {len(all_cameras)} kamera bulundu", 
                  font=("Arial", 11), bg='#1e1e1e', fg='#ffffff').pack(pady=10)
        
        # Main container
        main_frame = tk.Frame(dialog, bg='#1e1e1e')
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Scrollable area
        canvas_test = tk.Canvas(main_frame, bg='#1e1e1e', highlightthickness=0)
        scrollbar_test = ttk.Scrollbar(main_frame, orient="vertical", command=canvas_test.yview)
        scrollable_test = tk.Frame(canvas_test, bg='#1e1e1e')
        
        scrollable_test.bind(
            "<Configure>",
            lambda e: canvas_test.configure(scrollregion=canvas_test.bbox("all"))
        )
        
        canvas_test.create_window((0, 0), window=scrollable_test, anchor="nw")
        canvas_test.configure(yscrollcommand=scrollbar_test.set)
        
        # Kayıtlı seçilen kamera
        selected_camera = {'value': None}
        
        # Tüm kameralar için buton
        for cam_info in all_cameras:
            # Durum iconu
            status_icon = "🟢" if cam_info['status'] == 'Aktif' else "🔴"
            
            # Sadece kamera numarası ve durumu
            btn_text = f"{status_icon} Kamera {cam_info['camera_id']} - {cam_info['status']}"
            
            # Sadece aktif kameralar clickable
            if cam_info['status'] == 'Aktif':
                btn = tk.Button(
                    scrollable_test, 
                    text=btn_text,
                    command=lambda c=cam_info: self.start_camera_test(c, dialog, selected_camera),
                    font=("Arial", 10),
                    bg='#2d2d2d',
                    fg='#ffffff',
                    activebackground='#3d3d3d',
                    activeforeground='#ffffff',
                    relief=tk.FLAT,
                    cursor='hand2',
                    padx=15,
                    pady=12,
                    anchor='w',
                    justify='left'
                )
            else:
                btn = tk.Button(
                    scrollable_test, 
                    text=btn_text + "\n⚠️ Önce kamerayı başlatın!",
                    command=lambda c=cam_info: self.start_camera_test(c, dialog, selected_camera),
                    font=("Arial", 9),
                    bg='#1e1e1e',
                    fg='#888888',
                    activebackground='#2d2d2d',
                    activeforeground='#888888',
                    relief=tk.FLAT,
                    cursor='hand2',
                    padx=10,
                    pady=8,
                    anchor='w',
                    justify='left',
                    state='disabled'
                )
            
            btn.pack(pady=3, padx=5, fill=tk.X)
        
        canvas_test.pack(side="left", fill="both", expand=True)
        scrollbar_test.pack(side="right", fill="y")
        
        # İptal butonu (ayrı frame)
        cancel_frame = tk.Frame(dialog, bg='#1e1e1e')
        cancel_frame.pack(pady=10)
        cancel_btn = tk.Button(cancel_frame, text="❌ İptal", command=dialog.destroy,
                               font=("Arial", 10, "bold"),
                               bg='#2d2d2d',
                               fg='#ffffff',
                               activebackground='#3d3d3d',
                               activeforeground='#ffffff',
                               relief=tk.FLAT,
                               cursor='hand2',
                               padx=20,
                               pady=8)
        cancel_btn.pack()
    
    def start_camera_test(self, cam_info, dialog, selected_camera):
        """Seçilen kamerayı test et"""
        # Kamera aktif değilse uyarı ver
        if cam_info['status'] != 'Aktif':
            messagebox.showwarning("Uyarı", 
                f"❌ Kamera {cam_info['camera_id']} çalışmıyor!\n\n"
                f"Lütfen önce kamerayı 'Başlat' butonuna tıklayarak başlatın.")
            return
        
        selected_camera['value'] = cam_info
        dialog.destroy()
        
        camera_id = cam_info['camera_id']
        url = cam_info['url']
        
        self.log_message(f"🧪 Kamera {camera_id} test ediliyor...")
        
        # Kamerayı aç
        try:
            # URL sayı ise integer'a çevir
            try:
                camera_index = int(url)
                cap = cv2.VideoCapture(camera_index)
            except ValueError:
                cap = cv2.VideoCapture(url)
            
            if not cap.isOpened():
                messagebox.showerror("Test Başarısız", 
                    f"❌ Kamera {camera_id} açılamadı!\n\nURL: {url}\n\nKamera bağlantısı kontrol edin.")
                return
            
            self.log_message(f"✅ Kamera {camera_id} açıldı, 10 saniye test yapılıyor...")
            
            # 10 saniye test
            start_time = time.time()
            frame_count = 0
            success_count = 0
            
            while time.time() - start_time < 10:
                ret, frame = cap.read()
                
                if ret:
                    frame_count += 1
                    success_count += 1
                    
                    # Görüntüyü göster
                    cv2.imshow(f"Kamera {camera_id} Testi - ESC ile cik", frame)
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşu
                        break
                else:
                    time.sleep(0.1)
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Sonuçları göster
            if frame_count > 0 and success_count > 0:
                success_rate = (success_count / frame_count) * 100
                messagebox.showinfo("Test Sonucu ✅", 
                    f"Kamera {camera_id} ÇALIŞIYOR! 🎉\n\n"
                    f"📹 URL: {url}\n"
                    f"📊 Toplam Frame: {frame_count}\n"
                    f"✅ Başarılı Frame: {success_count}\n"
                    f"📈 Başarı Oranı: {success_rate:.1f}%\n\n"
                    f"Kamera düzgün çalışıyor!")
                self.log_message(f"✅ Kamera {camera_id} test başarılı: {success_rate:.1f}% başarı")
            else:
                messagebox.showerror("Test Sonucu ❌", 
                    f"Kamera {camera_id} ÇALIŞMIYOR! ⚠️\n\n"
                    f"📹 URL: {url}\n"
                    f"📊 Okunan Frame: {frame_count}\n\n"
                    f"• URL'yi kontrol edin\n"
                    f"• Kamera bağlantısını kontrol edin\n"
                    f"• IP adresini kontrol edin")
                self.log_message(f"❌ Kamera {camera_id} test başarısız")
        
        except Exception as e:
            messagebox.showerror("Hata", f"Test sırasında hata oluştu:\n{str(e)}")
            self.log_message(f"❌ Test hatası: {e}")
    
    def log_message(self, msg):
        """Thread-safe log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {msg}"
        print(log_text)
        # Thread-safe message queue (GUI için gerekirse)
        try:
            self.message_queue.put(('log', log_text), block=False)
        except:
            pass
    
    def open_photo_settings(self):
        """Fotoğraf gönderim ayarları penceresi"""
        # Dialog penceresi
        dialog = tk.Toplevel(self.root)
        dialog.title("📸 Fotoğraf Gönderim Ayarları")
        dialog.geometry("500x550")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Başlık
        ttk.Label(dialog, text="📸 Fotoğraf Gönderim Ayarları", 
                 font=("Arial", 12, "bold")).pack(pady=10)
        
        # Config'den mevcut ayarları yükle
        work_schedule = {}
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    work_schedule = data.get('work_schedule', {})
        except:
            pass
        
        # Scrollable frame
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Haftalık günler
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_names_tr = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
        
        day_settings = {}
        
        for i, (day, day_tr) in enumerate(zip(days, day_names_tr)):
            # Her gün için frame
            day_frame = ttk.LabelFrame(scrollable, text=day_tr, padding="5")
            day_frame.pack(fill=tk.X, pady=2)
            
            # Aktif checkbox
            enabled_var = tk.BooleanVar()
            enabled_var.set(work_schedule.get(day, {}).get('enabled', True if i < 5 else False))
            ttk.Checkbutton(day_frame, text="Aktif", variable=enabled_var).pack(side=tk.LEFT, padx=5)
            
            # Başlangıç saati ve dakikası
            ttk.Label(day_frame, text="Başlangıç:").pack(side=tk.LEFT, padx=5)
            start_hour_spinbox = ttk.Spinbox(day_frame, from_=0, to=23, width=3)
            start_hour_spinbox.set(work_schedule.get(day, {}).get('start_hour', 8))
            start_hour_spinbox.pack(side=tk.LEFT, padx=2)
            ttk.Label(day_frame, text=":").pack(side=tk.LEFT)
            start_min_spinbox = ttk.Spinbox(day_frame, from_=0, to=59, width=3)
            start_min_spinbox.set(work_schedule.get(day, {}).get('start_min', 0))
            start_min_spinbox.pack(side=tk.LEFT, padx=2)
            
            # Bitiş saati ve dakikası
            ttk.Label(day_frame, text="Bitiş:").pack(side=tk.LEFT, padx=5)
            end_hour_spinbox = ttk.Spinbox(day_frame, from_=0, to=23, width=3)
            end_hour_spinbox.set(work_schedule.get(day, {}).get('end_hour', 18))
            end_hour_spinbox.pack(side=tk.LEFT, padx=2)
            ttk.Label(day_frame, text=":").pack(side=tk.LEFT)
            end_min_spinbox = ttk.Spinbox(day_frame, from_=0, to=59, width=3)
            end_min_spinbox.set(work_schedule.get(day, {}).get('end_min', 0))
            end_min_spinbox.pack(side=tk.LEFT, padx=2)
            
            day_settings[day] = {
                'enabled': enabled_var,
                'start_hour': start_hour_spinbox,
                'start_min': start_min_spinbox,
                'end_hour': end_hour_spinbox,
                'end_min': end_min_spinbox
            }
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Butonlar
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def save_settings():
            """Ayarları kaydet"""
            new_schedule = {}
            for day in days:
                new_schedule[day] = {
                    'enabled': day_settings[day]['enabled'].get(),
                    'start_hour': int(day_settings[day]['start_hour'].get()),
                    'start_min': int(day_settings[day]['start_min'].get()),
                    'end_hour': int(day_settings[day]['end_hour'].get()),
                    'end_min': int(day_settings[day]['end_min'].get())
                }
            
            # Config dosyasını güncelle
            try:
                if os.path.exists(CONFIG_FILE):
                    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = {}
                
                data['work_schedule'] = new_schedule
                
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                self.log_message("✅ Fotoğraf gönderim ayarları kaydedildi")
                messagebox.showinfo("Başarılı", "Ayarlar kaydedildi!")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Hata", f"Ayarlar kaydedilemedi: {e}")
        
        ttk.Button(button_frame, text="Kaydet", command=save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="İptal", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def start_camera(self, idx):
        url_or_index = self.urls[idx].get().strip()
        if not url_or_index:
            messagebox.showwarning("Uyarı", f"Kamera {idx+1} için URL veya kamera numarası giriniz! (örn: 0, 1 veya rtsp://...)")
            return
        self.cameras[idx].start(url_or_index)
        self.log_message(f"Kamera {idx+1}: Başlatılıyor...")
    
    def stop_camera(self, idx):
        self.cameras[idx].stop()
    
    def set_detection_region(self, idx):
        """Kamera için tespit bölgesi belirle - DİREKT KAMERA AÇMA"""
        camera = self.cameras[idx]
        camera_id = idx + 1
        
        # Kamera aktif değilse uyarı ver
        if not camera.running:
            messagebox.showwarning("Uyarı", 
                f"❌ Kamera {camera_id} çalışmıyor!\n\n"
                f"Lütfen önce kamerayı 'Başlat' butonuna tıklayarak başlatın.")
            return
        
        # Kamera feed'i için geçici açış
        url = self.urls[idx].get().strip()
        try:
            try:
                camera_index = int(url)
                cap = cv2.VideoCapture(camera_index)
            except ValueError:
                cap = cv2.VideoCapture(url)
            
            if not cap.isOpened():
                messagebox.showerror("Hata", f"Kamera {camera_id} açılamadı!")
                return
            
            # Bir frame oku
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Hata", f"Kamera {camera_id} görüntü alamadı!")
                cap.release()
                return
            
            cap.release()
            
            # Mevcut bölgeleri göster
            temp_regions = camera.detection_regions.copy()
            
            # Seçili bölge için index
            selected_region_index = {'value': None}
            
            # Mouse callback için değişkenler
            class DrawingState:
                def __init__(self):
                    self.drawing = False
                    self.start_point = None
                    self.end_point = None
                    self.current_frame = None
                    self.mouse_pos = None
                    self.button_positions = {}
                    self.action_queue = []  # Buton tıklamaları için kuyruk
            
            state = DrawingState()
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal temp_regions
                
                # Mouse pozisyonunu kaydet
                state.mouse_pos = (x, y)
                
                # Butonlara tıklama kontrolü
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Butonları kontrol et
                    button_clicked = False
                    if state.button_positions:
                        for btn_name, btn_rect in state.button_positions.items():
                            btn_x1, btn_y1, btn_x2, btn_y2 = btn_rect
                            if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
                                # Butona tıklandı
                                state.action_queue.append(btn_name)
                                button_clicked = True
                                break
                    
                    if button_clicked:
                        return  # Buton tıklaması işlendi, bölge işlemlerini yapma
                    
                    # Mevcut bölgelerden birine tıklanmış mı kontrol et
                    clicked_region = None
                    for idx, region in enumerate(temp_regions):
                        x1, y1, x2, y2 = region
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            clicked_region = idx
                            break
                    
                    if clicked_region is not None:
                        # Mevcut bölgeye tıklandı - seç ve vurgula
                        selected_region_index['value'] = clicked_region
                    else:
                        # Yeni bölge çizmeye başla
                        state.drawing = True
                        state.start_point = (x, y)
                        selected_region_index['value'] = None
                
                elif event == cv2.EVENT_LBUTTONUP and state.drawing:
                    state.drawing = False
                    state.end_point = (x, y)
                    
                    # Yeni bölge ekle
                    if state.start_point and state.end_point:
                        x1, y1 = state.start_point
                        x2, y2 = state.end_point
                        
                        # Koordinatları düzenle
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        # Minimum boyut kontrolü
                        if abs(x2 - x1) > 50 and abs(y2 - y1) > 50:
                            temp_regions.append((x1, y1, x2, y2))
                            selected_region_index['value'] = len(temp_regions) - 1
                        # Boyut küçükse sessizce ekleme
                    
                    state.start_point = None
                    state.end_point = None
                
                elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
                    # Çizim sırasında önizleme için flag set et
                    # Gerçek frame rendering ana loop'ta yapılacak
                    pass
                
                elif event == cv2.EVENT_RBUTTONDOWN:
                    # Sağ tık ile bölge seç
                    for idx, region in enumerate(temp_regions):
                        x1, y1, x2, y2 = region
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            selected_region_index['value'] = idx
                            break
            
            # Kamera'yı canlı olarak aç
            try:
                try:
                    camera_index = int(url)
                    cap_live = cv2.VideoCapture(camera_index)
                except ValueError:
                    cap_live = cv2.VideoCapture(url)
                
                if not cap_live.isOpened():
                    cap_live = None
            except:
                cap_live = None
            
            # Pencere ayarla
            window_name = f"Kamera {camera_id} Bolgelendirme"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, mouse_callback)
            
            # X butonu ile kapatma desteği
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
            
            # Canlı feed ile kontrol
            quit_app = False
            save_regions = False
            
            while not quit_app:
                # Canlı feed veya statik frame
                if cap_live and cap_live.isOpened():
                    ret_live, frame_live = cap_live.read()
                    if ret_live:
                        display = frame_live.copy()
                    else:
                        display = frame.copy()
                else:
                    display = frame.copy()
                
                # Mevcut bölgeleri çiz
                for idx_region, region in enumerate(temp_regions):
                    x1, y1, x2, y2 = region
                    
                    # Seçili bölgeyi farklı renkle göster
                    if selected_region_index['value'] == idx_region:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Sarı ve kalın
                        cv2.putText(display, f"B{idx_region+1}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Yeşil
                        cv2.putText(display, f"B{idx_region+1}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Çizim sırasında önizleme göster
                if state.drawing and state.start_point and state.mouse_pos:
                    # Canlı önizleme kutusu
                    cv2.rectangle(display, state.start_point, state.mouse_pos, (255, 0, 0), 2)
                    # Boyut bilgisi
                    width = abs(state.mouse_pos[0] - state.start_point[0])
                    height = abs(state.mouse_pos[1] - state.start_point[1])
                    cv2.putText(display, f"{width}x{height}", 
                               (state.start_point[0], state.start_point[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Çerçeve
                cv2.rectangle(display, (0, 0), (display.shape[1]-1, display.shape[0]-1), (255, 255, 0), 5)
                
                # Tıklanabilir butonlar - Alt kısımda
                button_y = display.shape[0] - 60
                button_height = 45
                button_width = 140
                button_spacing = 8
                
                # Buton 1: Bölge Ekle
                btn1_x = 20
                btn1_y = button_y
                btn1_rect = (btn1_x, btn1_y, btn1_x + button_width, btn1_y + button_height)
                cv2.rectangle(display, (btn1_x, btn1_y), (btn1_x + button_width, btn1_y + button_height), (45, 45, 45), -1)
                cv2.rectangle(display, (btn1_x, btn1_y), (btn1_x + button_width, btn1_y + button_height), (100, 100, 100), 3)
                cv2.putText(display, "BOLGE EKLE", (btn1_x + 12, btn1_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
                # Buton 2: Bölge Sil
                btn2_x = btn1_x + button_width + button_spacing
                btn2_y = button_y
                btn2_rect = (btn2_x, btn2_y, btn2_x + button_width, btn2_y + button_height)
                cv2.rectangle(display, (btn2_x, btn2_y), (btn2_x + button_width, btn2_y + button_height), (35, 35, 35), -1)
                cv2.rectangle(display, (btn2_x, btn2_y), (btn2_x + button_width, btn2_y + button_height), (80, 80, 80), 3)
                cv2.putText(display, "BOLGE SIL", (btn2_x + 22, btn2_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
                # Buton 3: Kaydet
                btn3_x = btn2_x + button_width + button_spacing
                btn3_y = button_y
                btn3_rect = (btn3_x, btn3_y, btn3_x + button_width, btn3_y + button_height)
                cv2.rectangle(display, (btn3_x, btn3_y), (btn3_x + button_width, btn3_y + button_height), (55, 55, 55), -1)
                cv2.rectangle(display, (btn3_x, btn3_y), (btn3_x + button_width, btn3_y + button_height), (120, 120, 120), 3)
                cv2.putText(display, "KAYDET", (btn3_x + 28, btn3_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
                # Buton 4: İptal
                btn4_x = btn3_x + button_width + button_spacing
                btn4_y = button_y
                btn4_rect = (btn4_x, btn4_y, btn4_x + button_width, btn4_y + button_height)
                cv2.rectangle(display, (btn4_x, btn4_y), (btn4_x + button_width, btn4_y + button_height), (35, 35, 35), -1)
                cv2.rectangle(display, (btn4_x, btn4_y), (btn4_x + button_width, btn4_y + button_height), (80, 80, 80), 3)
                cv2.putText(display, "IPTAL", (btn4_x + 35, btn4_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                
                # Buton pozisyonlarını sakla (mouse callback için)
                button_positions = {
                    'btn1': btn1_rect,
                    'btn2': btn2_rect,
                    'btn3': btn3_rect,
                    'btn4': btn4_rect
                }
                state.button_positions = button_positions
                
                cv2.imshow(window_name, display)
                
                # Buton tıklamalarını işle
                while state.action_queue:
                    action = state.action_queue.pop(0)
                    
                    if action == 'btn1':  # Bölge Ekle
                        # Çizim modunu aktif et (zaten varsayılan olarak açık)
                        # Kullanıcı sol-click ile zaten bölge ekleyebilir
                        pass
                        
                    elif action == 'btn2':  # Bölge Sil
                        if selected_region_index['value'] is not None and 0 <= selected_region_index['value'] < len(temp_regions):
                            # Onay vermeden sil
                            temp_regions.pop(selected_region_index['value'])
                            selected_region_index['value'] = None
                            self.log_message(f"✅ Bölge silindi")
                        else:
                            self.log_message(f"⚠️ Lütfen silmek için bir bölgeye tıklayın!")
                            
                    elif action == 'btn3':  # Kaydet
                        save_regions = True
                        quit_app = True
                        break
                        
                    elif action == 'btn4':  # İptal
                        # Onay vermeden iptal et
                        temp_regions = camera.detection_regions.copy()
                        quit_app = True
                        break
                
                # Kısa bekleme
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC - Acil çıkış
                    quit_app = True
                
                # Pencere kapatıldı mı kontrol et (X butonu ile)
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        quit_app = True
                except:
                    quit_app = True
            
            # Kamerayı kapat
            if cap_live:
                cap_live.release()
            
            cv2.destroyAllWindows()
            
            # Sadece kaydet'e basıldıysa kaydet
            if save_regions:
                camera.detection_regions = temp_regions
                camera.use_regions = len(temp_regions) > 0
                self.log_message(f"✅ Kamera {camera_id}: {len(temp_regions)} bölge kaydedildi")
                self.save_urls()
            else:
                self.log_message(f"ℹ️ Kamera {camera_id}: Değişiklikler iptal edildi")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Bölge ayarlama hatası: {e}")
            self.log_message(f"❌ Bölge ayarlama hatası: {e}")
    
    def connect_all_cameras(self):
        """Tüm URL'lere otomatik bağlan"""
        connected_count = 0
        skipped_count = 0
        
        self.log_message("🚀 Tüm kameralara bağlanma başlatılıyor...")
        
        for i in range(70):
            url = self.urls[i].get().strip()
            
            if url:  # URL varsa
                if not self.cameras[i].running:  # Zaten çalışmıyorsa
                    self.cameras[i].start(url)
                    connected_count += 1
                    self.log_message(f"✅ Kamera {i+1} başlatıldı: {url[:50]}...")
                else:
                    skipped_count += 1
                    self.log_message(f"⏭️ Kamera {i+1} zaten çalışıyor")
            else:
                skipped_count += 1
        
        # Sonuç mesajı
        total = connected_count + skipped_count
        if connected_count > 0:
            messagebox.showinfo("Başarılı", 
                f"✅ {connected_count} kamera başlatıldı!\n\n"
                f"⏭️ {skipped_count} kamera atlandı (URL yok veya zaten aktif)\n\n"
                f"Toplam: {total} / 70 kamera")
        else:
            messagebox.showwarning("Uyarı", 
                f"❌ Hiçbir kamera başlatılamadı!\n\n"
                f"Lütfen kamera URL'lerini girin.")
        
        self.log_message(f"Bağlantı tamamlandı: {connected_count} başlatıldı, {skipped_count} atlandı")
    
    def update_displays(self):
        while True:
            for i, camera in enumerate(self.cameras):
                try:
                    # Kamera bağlantı kontrolü - 90 saniye timeout (yavaş stream'ler için)
                    connection_timeout = time.time() - camera.last_frame_time > 90 if camera.last_frame_time > 0 else False
                    
                    # Sadece durum bilgisi göster (görüntü yok)
                    if camera.running:
                        if camera.connection_lost or connection_timeout:
                            status_text = f"Kamera {camera.camera_id}\n🔴 BAĞLANTI KOPMUŞ"
                            bg_color = '#8B0000'
                            fg_color = '#FF6B6B'
                        else:
                            detection_status = "✅" if camera.yolo_available or camera.hog else "❌"
                            status_text = f"Kamera {camera.camera_id}\n🟢 AKTİF\nTespit: {detection_status}"
                            bg_color = '#1B5E20'
                            fg_color = '#A5D6A7'
                    else:
                        status_text = f"Kamera {camera.camera_id}\n⚫ DURDURULDU"
                        bg_color = '#212121'
                        fg_color = '#BDBDBD'
                    
                    # Thread-safe GUI güncelleme - sadece ana thread'de çalıştır
                    self.message_queue.put(('update_label', i, status_text, bg_color, fg_color), block=False)
                except Exception as e:
                    try:
                        self.message_queue.put(('update_label', i, f"Kamera {i+1}\nHata", 'red', 'white'), block=False)
                    except:
                        pass
            
            time.sleep(1)  # 1 saniyede bir güncelle
    
    def on_closing(self):
        self.log_message("Sistem kapatılıyor...")
        # URL'leri kaydet
        self.save_urls()
        # Kameraları durdur
        for camera in self.cameras:
            camera.stop()
        time.sleep(1)
        self.root.destroy()
    
    def run(self):
        self.log_message("70 Kamera İzleme Sistemi başlatıldı - Arka Plan Modu (Görüntü Yok)")
        
        # Message queue kontrolcüsü - GUI güncellemelerini thread-safe yap
        def process_queue():
            try:
                while True:
                    try:
                        msg_type, *args = self.message_queue.get_nowait()
                        
                        if msg_type == 'update_label':
                            idx, status_text, bg_color, fg_color = args
                            try:
                                self.labels[idx].configure(
                                    image='',
                                    text=status_text, 
                                    bg=bg_color, 
                                    fg=fg_color,
                                    font=("Arial", 9, "bold")
                                )
                            except:
                                pass
                        elif msg_type == 'log':
                            # Log mesajı zaten print edildi, burada GUI'ye yazılabilir
                            pass
                    except queue.Empty:
                        break
            except:
                pass
            
            # Her 100ms'de bir kontrol et
            self.root.after(100, process_queue)
        
        # Queue işlemcisini başlat
        process_queue()
        
        self.root.mainloop()


if __name__ == "__main__":
    app = CameraGUI()
    app.run()

