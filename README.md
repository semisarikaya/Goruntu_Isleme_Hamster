# 🐹 HamsterGallery - El & Yüz İfadesiyle Kontrol Edilen Galeri

**BL242 Görüntü İşleme - Final Projesi**  
**Dönem:** 2025-2026 Bahar  

---

## 📌 Proje Hakkında

Gerçek zamanlı kamera görüntüsü ile çalışan, el hareketleri ve yüz ifadelerine göre farklı hamster fotoğrafları gösteren interaktif bir galeri uygulamasıdır.

- **El algılanırsa** → el hareketine göre hamster değişir
- **El yoksa** → yüz ifadesine göre hamster değişir

---

## 🎮 Kontroller

| Hareket | Hamster |
|---|---|
| 🖐️ İki eli havada | İki eli havada hamster |
| ✌️ İki parmak | İki parmak gösteren hamster |
| 👍 Baş parmak | Baş parmak gösteren hamster |
| 😠 Kızgın | Kızgın hamster |
| 😊 Mutlu | Mutlu hamster |
| 😐 Normal | Normal hamster |
| 😲 Şaşıran | Şaşıran hamster |
| 😴 Uyuyan | Uyuyan hamster |
| 😢 Üzgün | Üzgün hamster |
| 🐹 Yanak şişkin | Yanakları dolu hamster |

---

## 🛠️ Kullanılan Teknolojiler

- Python 3.11
- OpenCV
- MediaPipe (Hand Landmarker, Face Landmarker)
- scikit-learn (Random Forest Classifier)
- NumPy, Pandas

---

## ⚙️ Kurulum

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn matplotlib seaborn
```

---

## 🚀 Çalıştırma

```bash
# 1. Veri topla
python 1_veri_topla.py

# 2. Modeli eğit
python 2_model_egit.py

# 3. Canlı test
python 3_canli_test.py
```

---

## 📊 Model Performansı

| Model | Test Doğruluğu | CV Doğruluğu |
|---|---|---|
| El Hareketi | %100 | %96.5 |
| Yüz İfadesi | %99.4 | %81.6 |
