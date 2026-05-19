"""
=============================================================
  HamsterGallery – Adım 3: Canlı Test (v2)
=============================================================
  Sol: Hamster fotoğrafı
  Sağ: Kamera

  el_model.pkl  → el hareketleri
  yuz_model.pkl → yüz ifadeleri
=============================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pickle
import os
import urllib.request
from collections import deque

# ─────────────────────────────────────────────
#  Model dosyaları
# ─────────────────────────────────────────────
HAND_MODEL  = "hand_landmarker.task"
FACE_MODEL  = "face_landmarker.task"
EL_MODEL    = "el_model.pkl"
YUZ_MODEL   = "yuz_model.pkl"
IMAGES_DIR  = "images"

def download_model(url, path):
    if not os.path.exists(path):
        print(f"  Indiriliyor: {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  Tamam: {path}")

download_model("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", HAND_MODEL)
download_model("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", FACE_MODEL)

# ─────────────────────────────────────────────
#  Modelleri yükle
# ─────────────────────────────────────────────
with open(EL_MODEL, "rb") as f:
    el_data = pickle.load(f)
el_clf = el_data["model"]
le_el  = el_data["label_encoder"]

with open(YUZ_MODEL, "rb") as f:
    yuz_data = pickle.load(f)
yuz_clf = yuz_data["model"]
le_yuz  = yuz_data["label_encoder"]

print(f"  El siniflar : {list(le_el.classes_)}")
print(f"  Yuz siniflar: {list(le_yuz.classes_)}")

# ─────────────────────────────────────────────
#  Fotoğraf eşleştirmesi
# ─────────────────────────────────────────────
LABEL_TO_FILE = {
    "iki_el"       : "iki_eli_havada.jpeg",
    "iki_parmak"   : "iki_parmak.jpeg",
    "bas_parmak"   : "bas_parmak.jpeg",
    "kizgin"       : "kizgin.jpeg",
    "mutlu"        : "mutlu.jpeg",
    "normal"       : "normal.jpeg",
    "sasiran"      : "sasiran.jpeg",
    "uyuyan"       : "uyuyan.jpeg",
    "uzgun"        : "uzgun.jpeg",
    "yanak_siskin" : "yanaklar_siskin.jpeg",
}

LABEL_TR = {
    "iki_el"       : "IKI EL HAVADA",
    "iki_parmak"   : "IKI PARMAK",
    "bas_parmak"   : "BAS PARMAK",
    "kizgin"       : "KIZGIN",
    "mutlu"        : "MUTLU",
    "normal"       : "NORMAL",
    "sasiran"      : "SASIRAN",
    "uyuyan"       : "UYUYAN",
    "uzgun"        : "UZGUN",
    "yanak_siskin" : "YANAK SISKIN",
}

LABEL_COLOR = {
    "iki_el"       : (0,230,120),
    "iki_parmak"   : (0,200,255),
    "bas_parmak"   : (50,255,50),
    "kizgin"       : (0,0,230),
    "mutlu"        : (0,200,100),
    "normal"       : (180,180,180),
    "sasiran"      : (0,200,255),
    "uyuyan"       : (150,100,255),
    "uzgun"        : (100,100,230),
    "yanak_siskin" : (0,180,230),
}

hamster_images = {}
for label, fname in LABEL_TO_FILE.items():
    path = os.path.join(IMAGES_DIR, fname)
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            hamster_images[label] = img
            print(f"  Fotograf: {label} OK")
        else:
            print(f"  [UYARI] Okunamadi: {path}")
    else:
        print(f"  [UYARI] Bulunamadi: {path}")

# ─────────────────────────────────────────────
#  MediaPipe
# ─────────────────────────────────────────────
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = mp_vision.HandLandmarker.create_from_options(hand_options)

face_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detector = mp_vision.FaceLandmarker.create_from_options(face_options)

# ─────────────────────────────────────────────
#  Feature çıkarma
# ─────────────────────────────────────────────
def extract_hand_features(result, expect_two_hands=False):
    if not result.hand_landmarks: return None
    if expect_two_hands and len(result.hand_landmarks) < 2: return None
    hands_to_use = result.hand_landmarks[:2] if expect_two_hands else result.hand_landmarks[:1]
    features = []
    for hand_lm in hands_to_use:
        wrist = hand_lm[0]
        for lm in hand_lm:
            features.extend([lm.x-wrist.x, lm.y-wrist.y, lm.z-wrist.z])
    expected = 126 if expect_two_hands else 63
    if len(features) < expected:
        features.extend([0.0]*(expected-len(features)))
    return features

def extract_face_features(result):
    if not result.face_landmarks: return None
    lm = result.face_landmarks[0]
    KEY_POINTS = [
        61,291,13,14,78,308,95,324,0,17,
        33,160,158,133,153,144,362,385,387,263,373,380,
        46,53,52,55,65,276,283,282,285,295,
        1,2,98,327,
        123,50,36,206,205,187,352,280,266,426,425,411,
    ]
    nose=lm[1]; left_e=lm[33]; right_e=lm[263]
    interoc=max(abs(left_e.x-right_e.x),1e-6)
    features=[]
    for idx in KEY_POINTS:
        p=lm[idx]
        features.append((p.x-nose.x)/interoc)
        features.append((p.y-nose.y)/interoc)
    return features

# ─────────────────────────────────────────────
#  Çizim
# ─────────────────────────────────────────────
def put_text(img,text,pos,scale=0.8,color=(255,255,255),thickness=2):
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_DUPLEX,scale,(0,0,0),thickness+2,cv2.LINE_AA)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_DUPLEX,scale,color,thickness,cv2.LINE_AA)

def make_panel(label, panel_w, panel_h):
    if label in hamster_images:
        img = cv2.resize(hamster_images[label].copy(), (panel_w, panel_h))
    else:
        color = LABEL_COLOR.get(label,(100,100,100))
        img = np.full((panel_h,panel_w,3), color, dtype=np.uint8)
    overlay = img.copy()
    cv2.rectangle(overlay,(0,panel_h-70),(panel_w,panel_h),(0,0,0),-1)
    cv2.addWeighted(overlay,0.6,img,0.4,0,img)
    ltext = LABEL_TR.get(label, label.upper())
    lcolor = LABEL_COLOR.get(label,(255,255,255))
    put_text(img, ltext, (15,panel_h-25), scale=1.1, color=lcolor, thickness=2)
    return img

def draw_hud(frame, label, conf, mod, hand_count):
    h,w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(w,100),(15,15,15),-1)
    cv2.addWeighted(overlay,0.75,frame,0.25,0,frame)
    mod_text  = "EL MODU" if mod=="hand" else "YUZ MODU"
    mod_color = (0,230,120) if mod=="hand" else (230,150,0)
    put_text(frame, mod_text, (15,32), scale=0.8, color=mod_color)
    ltext  = LABEL_TR.get(label,label.upper())
    lcolor = LABEL_COLOR.get(label,(255,255,255))
    put_text(frame, f"Tahmin: {ltext}", (15,65), scale=0.8, color=lcolor)
    conf_color = (0,230,100) if conf>80 else (0,180,255) if conf>60 else (0,100,230)
    put_text(frame, f"Guven: %{conf:.0f}", (w-220,32), scale=0.75, color=conf_color)
    if mod=="hand":
        put_text(frame, f"El: {hand_count}", (w-220,65), scale=0.75, color=(200,200,200))
    put_text(frame, "Q: Cikis", (15,h-12), scale=0.5, color=(140,140,140), thickness=1)

# ─────────────────────────────────────────────
#  Smoothing
# ─────────────────────────────────────────────
SMOOTH_N = 6
pred_history = deque(maxlen=SMOOTH_N)

def smooth_predict(new_label):
    pred_history.append(new_label)
    return max(set(pred_history), key=pred_history.count)

# ─────────────────────────────────────────────
#  Ana döngü
# ─────────────────────────────────────────────
def main():
    print("\n"+"="*55)
    print("  HamsterGallery - Canli Test v2")
    print("="*55)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı!"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    PANEL_W, PANEL_H = 480, cam_h

    current_label = "normal"
    current_conf  = 0.0
    print("  Hazir! Q ile cikis.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = hand_detector.detect(mp_image)
        hand_count  = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        mod = "face"
        pred_label = current_label
        confidence = current_conf

        if hand_count > 0:
            mod = "hand"
            # Önce iki el dene, olmadı tek el
            features = extract_hand_features(hand_result, expect_two_hands=(hand_count>=2))
            if features is None:
                features = extract_hand_features(hand_result, expect_two_hands=False)

            if features is not None:
                feat = np.nan_to_num(np.array(features).reshape(1,-1))
                # Boyut uyarla
                exp = el_clf.n_features_in_
                if feat.shape[1] < exp:
                    feat = np.hstack([feat, np.zeros((1,exp-feat.shape[1]))])
                elif feat.shape[1] > exp:
                    feat = feat[:,:exp]
                proba = el_clf.predict_proba(feat)[0]
                idx   = np.argmax(proba)
                pred_label = le_el.inverse_transform([idx])[0]
                confidence = proba[idx]*100

        else:
            face_result = face_detector.detect(mp_image)
            if face_result.face_landmarks:
                features = extract_face_features(face_result)
                if features is not None:
                    feat = np.nan_to_num(np.array(features).reshape(1,-1))
                    exp = yuz_clf.n_features_in_
                    if feat.shape[1] < exp:
                        feat = np.hstack([feat, np.zeros((1,exp-feat.shape[1]))])
                    elif feat.shape[1] > exp:
                        feat = feat[:,:exp]
                    proba = yuz_clf.predict_proba(feat)[0]
                    idx   = np.argmax(proba)
                    pred_label = le_yuz.inverse_transform([idx])[0]
                    confidence = proba[idx]*100

        if confidence > 30:
            smoothed = smooth_predict(pred_label)
            current_label = smoothed
            current_conf  = confidence

        panel = make_panel(current_label, PANEL_W, PANEL_H)
        draw_hud(frame, current_label, current_conf, mod, hand_count)
        frame_r = cv2.resize(frame, (cam_w, PANEL_H))
        combined = np.hstack([panel, frame_r])
        cv2.imshow("HamsterGallery", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break

    cap.release()
    hand_detector.close()
    face_detector.close()
    cv2.destroyAllWindows()
    print("  Gorüsürüz!")

if __name__ == "__main__":
    main()
