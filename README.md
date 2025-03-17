# Flower Predictor

Bu proje, bir çiçek görüntüsünü analiz ederek türünü tahmin eden bir derin öğrenme modeli ve Flask tabanlı bir web sunucusunu içerir.

## 📌 Özellikler
- **Transfer Learning**: MobileNetV2 modeli kullanılarak eğitildi.
- **Veri İşleme**: Veri artırma (augmentation) teknikleri ile modelin doğruluğu artırıldı.
- **Flask API**: Kullanıcıların bir çiçek resmi yükleyerek tahmin almasını sağlayan bir web arayüzü içerir.
- **Top 3 Tahmin**: Model, en yüksek olasılığa sahip 3 çiçek türünü ve bunların yüzdelik olasılıklarını döndürür.

## 🚀 Kurulum
Projeyi çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

### 1️⃣ Depoyu Klonlayın
```bash
git clone git@github.com:KutayKoray/flower_predictor.git
cd flower_predictor
```

### 2️⃣ Gerekli Kütüphaneleri Yükleyin
Python'un bağımlılıklarını yüklemek için aşağıdaki komutu çalıştırın:
```bash
pip install -r requirements.txt
```

### 3️⃣ Modeli Eğitin (Opsiyonel)
Eğer kendi modelinizi eğitmek isterseniz:
```bash
python train.py
```
Bu işlem sonucunda `best_flower_model.h5` dosyası oluşturulacaktır.

### 4️⃣ Flask Sunucusunu Başlatın
```bash
python server.py
```
Sunucu başlatıldıktan sonra `http://127.0.0.1:5000/` adresinden erişebilirsiniz.

## 📸 Kullanım
1. Web arayüzüne giriş yapın.
2. Bir çiçek resmi yükleyin.
3. Modelin tahminlerini ve olasılıklarını görün.

Alternatif olarak, API üzerinden tahmin almak için:
```bash
curl -X POST -F "file=@path/to/image.jpg" http://127.0.0.1:5000/predict
```

## 📂 Proje Yapısı
```
flower_predictor/
│── flowers/                # Çiçek veri seti
│── static/uploads/         # Yüklenen görsellerin kaydedildiği klasör
│── train.py                # Modeli eğiten dosya
│── server.py               # Flask sunucusunu başlatan dosya
│── best_flower_model.h5    # Eğitilmiş model
│── requirements.txt        # Bağımlılıkların listesi
│── README.md               # Proje açıklaması
```

## 📜 Lisans
Bu proje MIT lisansı ile lisanslanmıştır.

---
📌 Geliştirme ve katkılarınız için PR göndermekten çekinmeyin! 🚀

