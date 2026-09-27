from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

# YOLO wird erst importiert, wenn die Datei geladen wird.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ---------------------------------------------------------
# Deine Kategorien aus app.py
# ---------------------------------------------------------

CLASSES = [
    "Kleidungsstücke",
    "Schulsachen",
    "Trinkflasche",
    "Brotdose",
    "Regenschirm",
    "Schlüssel",
    "Kopfhörer",
    "Powerbank/Ladekabel",
    "Brille",
    "Geldtasche",
    "Taschenrechner",
]


# ---------------------------------------------------------
# YOLO-Modell
# ---------------------------------------------------------

MODEL_PATH = Path("yolo11n.pt")

_model = None


def get_model():
    global _model

    if YOLO is None:
        return None

    if _model is None:
        # Lädt yolo11n.pt beim ersten Start automatisch herunter,
        # falls die Datei noch nicht vorhanden ist.
        _model = YOLO(str(MODEL_PATH))

    return _model


# ---------------------------------------------------------
# YOLO-Klassen -> Fundbox-Kategorien
# ---------------------------------------------------------

CATEGORY_MAP = {
    # Trinkflasche
    "bottle": "Trinkflasche",

    # Regenschirm
    "umbrella": "Regenschirm",

    # Schulsachen
    "book": "Schulsachen",
    "backpack": "Schulsachen",
    "laptop": "Schulsachen",
    "keyboard": "Schulsachen",
    "mouse": "Schulsachen",

    # Kleidungsstücke
    "tie": "Kleidungsstücke",
    "suitcase": "Kleidungsstücke",

    # Geldtasche
    "handbag": "Geldtasche",

    # Dinge, die YOLO als Handy erkennt
    # können in der Fundbox nicht sinnvoll einer
    # eigenen Kategorie zugeordnet werden.
}


# Anzeigenamen für deine Fundbox
DISPLAY_NAMES = {
    "Kleidungsstücke": "Kleidungsstück",
    "Schulsachen": "Schulsache",
    "Trinkflasche": "Trinkflasche",
    "Brotdose": "Brotdose",
    "Regenschirm": "Regenschirm",
    "Schlüssel": "Schlüssel",
    "Kopfhörer": "Kopfhörer",
    "Powerbank/Ladekabel": "Powerbank/Ladekabel",
    "Brille": "Brille",
    "Geldtasche": "Geldtasche",
    "Taschenrechner": "Taschenrechner",
}


# ---------------------------------------------------------
# Hilfsfunktion
# ---------------------------------------------------------

def _empty_result() -> dict[str, Any]:
    return {
        "label": "unbekannt",
        "category": "Schulsachen",
        "confidence": 0.0,
        "top3": [],
        "engine": "YOLO11n",
    }


# ---------------------------------------------------------
# Hauptfunktion für deine app.py
# ---------------------------------------------------------

def predict_teachable(image: Image.Image) -> dict[str, Any] | None:
    """
    Kompatibel mit deinem bestehenden app.py.

    Eingabe:
        PIL.Image

    Ausgabe:
        {
            "label": ...,
            "category": ...,
            "confidence": ...,
            "top3": [...],
            "engine": ...
        }
    """

    model = get_model()

    if model is None:
        return None

    try:
        image = image.convert("RGB")

        results = model.predict(
            source=image,
            conf=0.15,
            imgsz=640,
            verbose=False,
            max_det=10,
        )

        if not results:
            return None

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        names = result.names

        detections = []

        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            confidence = float(result.boxes.conf[i].item())

            raw_label = names.get(cls_id, str(cls_id)).lower()

            category = CATEGORY_MAP.get(raw_label)

            if category is None:
                continue

            detections.append(
                {
                    "raw": raw_label,
                    "category": category,
                    "confidence": confidence,
                }
            )

        if not detections:
            return None

        # Höchste Sicherheit zuerst
        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )

        best = detections[0]

        # Mehrere YOLO-Erkennungen derselben Kategorie
        # werden zusammengeführt.
        category_scores = {}

        for detection in detections:
            category = detection["category"]
            score = detection["confidence"]

            if category not in category_scores:
                category_scores[category] = score
            else:
                category_scores[category] = max(
                    category_scores[category],
                    score
                )

        sorted_categories = sorted(
            category_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top3 = [
            (category, score)
            for category, score in sorted_categories[:3]
        ]

        best_category = sorted_categories[0][0]
        best_confidence = sorted_categories[0][1]

        label = DISPLAY_NAMES.get(
            best_category,
            best_category
        )

        return {
            "label": label,
            "category": best_category,
            "confidence": best_confidence,
            "top3": top3,
            "engine": "YOLO11n · COCO",
        }

    except Exception as e:
        print(f"YOLO-Fehler: {e}")
        return None


# ---------------------------------------------------------
# Deine alte Funktion bleibt kompatibel
# ---------------------------------------------------------

def heuristic_guess(image: Image.Image) -> dict[str, Any]:
    """
    Fallback, falls YOLO nichts erkennt.
    """

    return {
        "label": "unbekannt",
        "category": "Schulsachen",
        "confidence": 0.0,
        "top3": [],
        "engine": "Fallback",
    }


# ---------------------------------------------------------
# Kompatibilität mit deinem bisherigen Code
# ---------------------------------------------------------

def load_teachable_labels():
    return CLASSES
