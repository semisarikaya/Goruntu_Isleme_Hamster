"""
=============================================================
  HamsterGallery – Adım 1: Veri Toplama (v2)
=============================================================
  El verisi  → el_data.csv   (63 veya 126 boyut)
  Yüz verisi → yuz_data.csv  (100 boyut)
=============================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pandas as pd
import os
import time
import urllib.request

HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

def download_model(url, path):
    if not os.path.exists(path):
        print(f"  Indiriliyor: {path} ...")
        urllib.request.urlretrieve(url, path)
        print(f"  Tamam: {path}")

download_model("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", HAND_MODEL)
download_model("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task", FACE_MODEL)

SAMPLES_PER_CLASS = 500
EL_CSV  = "el_data.csv"
YUZ_CSV = "yuz_data.csv"
COUNTDOWN_SEC = 3

EL_KATEGORILER = [
    {"label": "iki_el",     "iki_el": True},
    {"label": "iki_parmak", "iki_el": False},
    {"label": "bas_parmak", "iki_el": False},
]

YUZ_KATEGORILER = [
    "kizgin", "mutlu", "normal", "sasiran", "uyuyan", "uzgun", "yanak_siskin",
]

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

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]

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

def put_text(img,text,pos,scale=0.8,color=(255,255,255),thickness=2):
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_DUPLEX,scale,(0,0,0),thickness+2,cv2.LINE_AA)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_DUPLEX,scale,color,thickness,cv2.LINE_AA)

def draw_overlay(img,text_lines,bar_ratio=0.0):
    w=img.shape[1]
    overlay=img.copy()
    cv2.rectangle(overlay,(0,0),(w,110),(20,20,20),-1)
    cv2.addWeighted(overlay,0.7,img,0.3,0,img)
    for i,(txt,col) in enumerate(text_lines):
        put_text(img,txt,(15,32+i*32),scale=0.75,color=col)
    if bar_ratio>0:
        bar_w=int((w-30)*bar_ratio)
        cv2.rectangle(img,(15,90),(w-15,106),(60,60,60),-1)
        cv2.rectangle(img,(15,90),(15+bar_w,106),(0,200,100),-1)

def draw_hand(frame,result):
    if not result.hand_landmarks: return
    h,w=frame.shape[:2]
    for hand_lm in result.hand_landmarks:
        pts=[(int(lm.x*w),int(lm.y*h)) for lm in hand_lm]
        for a,b in HAND_CONNECTIONS: cv2.line(frame,pts[a],pts[b],(0,200,255),2)
        for p in pts: cv2.circle(frame,p,5,(0,255,180),-1)

def draw_face(frame,result):
    if not result.face_landmarks: return
    h,w=frame.shape[:2]
    KEY=[61,291,13,14,33,133,362,263,1,55,285,159,145,386,374]
    for face_lm in result.face_landmarks:
        for idx in KEY:
            lm=face_lm[idx]
            cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),3,(0,255,180),-1)

def collect_category(cap,label,mod,iki_el,existing_done):
    if label in existing_done:
        print(f"  '{label}' zaten tamamlandi, atlaniyor.")
        return None
    print(f"\n  Kategori: {label}  |  Mod: {mod}")
    while True:
        ret,frame=cap.read()
        if not ret: return None
        frame=cv2.flip(frame,1)
        draw_overlay(frame,[
            (f"Sonraki: {label.upper()}",(100,220,255)),
            (f"{SAMPLES_PER_CLASS} ornek toplanacak",(200,200,200)),
            ("SPACE = Baslat   |   Q = Cikis",(180,180,100)),
        ])
        put_text(frame,label.upper(),(frame.shape[1]//2-180,frame.shape[0]//2),scale=1.4,color=(0,255,180))
        cv2.imshow("Veri Toplama",frame)
        key=cv2.waitKey(1)&0xFF
        if key==ord(' '): break
        if key in [ord('q'),27]: return "QUIT"
    for i in range(COUNTDOWN_SEC,0,-1):
        t0=time.time()
        while time.time()-t0<1.0:
            ret,frame=cap.read()
            if not ret: break
            frame=cv2.flip(frame,1)
            draw_overlay(frame,[(f"Kategori: {label}",(100,220,255)),(f"Basliyor: {i}...",(50,255,50))])
            put_text(frame,str(i),(frame.shape[1]//2-30,frame.shape[0]//2),scale=4.0,color=(0,255,100),thickness=6)
            cv2.imshow("Veri Toplama",frame)
            cv2.waitKey(1)
    records=[]
    collected=0
    while collected<SAMPLES_PER_CLASS:
        ret,frame=cap.read()
        if not ret: break
        frame=cv2.flip(frame,1)
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
        features=None
        if mod=="hand":
            result=hand_detector.detect(mp_image)
            draw_hand(frame,result)
            features=extract_hand_features(result,expect_two_hands=iki_el)
        else:
            result=face_detector.detect(mp_image)
            draw_face(frame,result)
            features=extract_face_features(result)
        if features is not None:
            record={f"f{i}":v for i,v in enumerate(features)}
            record["label"]=label
            records.append(record)
            collected+=1
            flash=frame.copy()
            cv2.rectangle(flash,(0,0),(frame.shape[1],frame.shape[0]),(0,255,100),8)
            cv2.addWeighted(flash,0.3,frame,0.7,0,frame)
        draw_overlay(frame,[
            (f"Kaydediliyor: {label}",(100,220,255)),
            (f"Ornek: {collected}/{SAMPLES_PER_CLASS}",(50,255,50)),
            ("Dogal ifade yap, cok abartma!",(200,200,100)),
        ],bar_ratio=collected/SAMPLES_PER_CLASS)
        cv2.imshow("Veri Toplama",frame)
        key=cv2.waitKey(1)&0xFF
        if key in [ord('q'),27]:
            return records if records else "QUIT"
    print(f"  '{label}' tamamlandi: {collected} ornek")
    return records

def _save(records,filename):
    if not records: return
    df=pd.DataFrame(records)
    df.to_csv(filename,index=False)
    print(f"  Kaydedildi: {filename} ({len(df)} satir)")
    print(df["label"].value_counts().to_string())

def main():
    print("\n"+"="*55)
    print("  HamsterGallery - Veri Toplama v2")
    print("  El  → el_data.csv")
    print("  Yuz → yuz_data.csv")
    print("="*55)
    el_records,yuz_records=[],[]
    el_done,yuz_done=set(),set()
    if os.path.exists(EL_CSV):
        df=pd.read_csv(EL_CSV)
        for lbl,cnt in df["label"].value_counts().items():
            if cnt>=SAMPLES_PER_CLASS: el_done.add(lbl)
        el_records=df.to_dict("records")
        print(f"  Mevcut el verisi: {len(df)} satir")
    if os.path.exists(YUZ_CSV):
        df=pd.read_csv(YUZ_CSV)
        for lbl,cnt in df["label"].value_counts().items():
            if cnt>=SAMPLES_PER_CLASS: yuz_done.add(lbl)
        yuz_records=df.to_dict("records")
        print(f"  Mevcut yuz verisi: {len(df)} satir")
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera açılamadı!"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    print("\n  === EL KATEGORİLERİ ===")
    for cat in EL_KATEGORILER:
        result=collect_category(cap,cat["label"],"hand",cat["iki_el"],el_done)
        if result=="QUIT": break
        if result: el_records.extend(result)
        _save(el_records,EL_CSV)
    print("\n  === YUZ KATEGORİLERİ ===")
    for label in YUZ_KATEGORILER:
        result=collect_category(cap,label,"face",False,yuz_done)
        if result=="QUIT": break
        if result: yuz_records.extend(result)
        _save(yuz_records,YUZ_CSV)
    cap.release()
    cv2.destroyAllWindows()
    _save(el_records,EL_CSV)
    _save(yuz_records,YUZ_CSV)

if __name__=="__main__":
    main()
