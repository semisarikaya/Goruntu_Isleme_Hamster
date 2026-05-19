"""
=============================================================
  HamsterGallery - Adim 2: Model Egitimi (v2)
=============================================================
  el_data.csv  -> el_model.pkl   (el hareketleri)
  yuz_data.csv -> yuz_model.pkl  (yuz ifadeleri)

  Kullanilan Algoritma: Random Forest Classifier
  Kutuphaneler: pandas, numpy, scikit-learn, matplotlib, seaborn

  Calistirma:
    python 2_model_egit.py
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Dosya yollari
EL_CSV    = "el_data.csv"
YUZ_CSV   = "yuz_data.csv"
EL_MODEL  = "el_model.pkl"
YUZ_MODEL = "yuz_model.pkl"


def egit_ve_kaydet(csv_path, model_path, model_adi):
    """
    CSV dosyasindan veri okur, Random Forest modeli egitir,
    confusion matrix cizer ve modeli kaydeder.
    """
    print("\n" + "="*55)
    print(f"  {model_adi} Egitimi Basliyor")
    print("="*55)

    # Veriyi yukle
    df = pd.read_csv(csv_path)
    print(f"\n  Toplam ornek  : {len(df)}")
    print(f"  Ozellik sayisi: {len(df.columns)-1}")
    print(f"\n  Sinif dagilimi:")
    print(df["label"].value_counts().to_string())

    # X ve y ayir
    X = np.nan_to_num(df.drop("label", axis=1).values)  # NaN -> 0
    y = df["label"].values

    # Label encoder: string -> sayi
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"\n  Siniflar: {list(le.classes_)}")

    # Train / Test bol (%80 / %20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc      # Her siniftan esit oranda al
    )
    print(f"\n  Egitim seti: {len(X_train)} ornek")
    print(f"  Test seti  : {len(X_test)} ornek")

    # Random Forest modeli egit
    print("\n  Model egitiliyor...")
    model = RandomForestClassifier(
        n_estimators=200,   # 200 karar agaci
        max_depth=20,       # Maksimum agac derinligi
        min_samples_split=4,
        random_state=42,
        n_jobs=-1           # Tum CPU cekirdeklerini kullan
    )
    model.fit(X_train, y_train)
    print("  Egitim tamamlandi!")

    # Test sonuclari
    y_pred   = model.predict(X_test)
    accuracy = (y_pred == y_test).mean() * 100
    print(f"\n  Test Dogrulugu: %{accuracy:.1f}")

    # 5 katli capraz dogrulama
    cv_scores = cross_val_score(model, X, y_enc, cv=5)
    print(f"  5-Fold CV    : %{cv_scores.mean()*100:.1f} +/- {cv_scores.std()*100:.1f}")

    print("\n  Sinif bazli sonuclar:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=le.classes_,
                yticklabels=le.classes_,
                linewidths=0.5)
    plt.title(f"{model_adi} Confusion Matrix  |  %{accuracy:.1f}",
              fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Gercek", fontsize=11)
    plt.xlabel("Tahmin", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    img_path = model_path.replace(".pkl", "_confusion.png")
    plt.savefig(img_path, dpi=150)
    plt.show()
    print(f"  Confusion matrix kaydedildi: {img_path}")

    # Modeli kaydet
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le}, f)
    print(f"  Model kaydedildi: {model_path}")

    return accuracy, cv_scores.mean()*100


def main():
    print("\n" + "="*55)
    print("  HamsterGallery - Model Egitimi v2")
    print("  El  modeli -> el_model.pkl")
    print("  Yuz modeli -> yuz_model.pkl")
    print("="*55)

    # CSV dosyalarini kontrol et
    for f in [EL_CSV, YUZ_CSV]:
        if not os.path.exists(f):
            print(f"  [HATA] {f} bulunamadi! Once 1_veri_topla.py calistir.")
            return

    # El modelini egit
    acc_el, cv_el = egit_ve_kaydet(EL_CSV, EL_MODEL, "El Hareketi")

    # Yuz modelini egit
    acc_yuz, cv_yuz = egit_ve_kaydet(YUZ_CSV, YUZ_MODEL, "Yuz Ifadesi")

    # Ozet
    print("\n" + "="*55)
    print("  OZET")
    print("="*55)
    print(f"  El  modeli -> Test: %{acc_el:.1f}   CV: %{cv_el:.1f}")
    print(f"  Yuz modeli -> Test: %{acc_yuz:.1f}  CV: %{cv_yuz:.1f}")
    print("\n  Adim 3: Canli test icin 3_canli_test.py calistir!")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
