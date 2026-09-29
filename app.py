```python
import datetime
import io
import json
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# SETUP
# ============================================================

st.set_page_config(
    page_title="Fundbox · Katharineum",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# DATEIEN
# ============================================================

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
ITEMS_FILE = DATA_DIR / "items.json"

DATA_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)


# ============================================================
# YOLO
# ============================================================

MODEL_FILE = Path("yolo11n.pt")


@st.cache_resource
def load_model():
    """
    Lädt das lokale YOLO11n-Modell.
    Die Datei yolo11n.pt muss im GitHub-Repository liegen.
    """
    return YOLO(str(MODEL_FILE))


model = load_model()


# ============================================================
# RELEVANTE YOLO-KATEGORIEN
# ============================================================

# Diese Kategorien existieren im normalen COCO-Modell
# von YOLO und sind für ein Fundbüro sinnvoll.
YOLO_CATEGORIES = {
    "backpack": "Backpack",
    "handbag": "Handbag",
    "suitcase": "Suitcase",
    "bottle": "Bottle",
    "cup": "Cup",
    "book": "Book",
    "laptop": "Laptop",
    "cell phone": "Cell Phone",
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "remote": "Remote",
    "clock": "Clock",
    "umbrella": "Umbrella",
    "sports ball": "Sports Ball",
    "tie": "Tie",
}

CLASS_EMOJI = {
    "Backpack": "🎒",
    "Handbag": "👜",
    "Suitcase": "🧳",
    "Bottle": "🥤",
    "Cup": "🥤",
    "Book": "📚",
    "Laptop": "💻",
    "Cell Phone": "📱",
    "Keyboard": "⌨️",
    "Mouse": "🖱️",
    "Remote": "🎮",
    "Clock": "⌚",
    "Umbrella": "☂️",
    "Sports Ball": "⚽",
    "Tie": "👔",
    "Unknown": "❓",
}

CLASSES = list(YOLO_CATEGORIES.values())


# ============================================================
# STORAGE
# ============================================================

def load_items():
    if ITEMS_FILE.exists():
        try:
            return json.loads(
                ITEMS_FILE.read_text(encoding="utf-8")
            )
        except Exception:
            return []

    return []


def save_items(items):
    ITEMS_FILE.write_text(
        json.dumps(
            items,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


if "items" not in st.session_state:
    st.session_state["items"] = load_items()

items = st.session_state["items"]


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("tab", "Browse")
st.session_state.setdefault("rep_bytes", None)
st.session_state.setdefault("rep_ai", None)
st.session_state.setdefault("rep_name_ai", "")
st.session_state.setdefault("uploader_nonce", 0)
st.session_state.setdefault("search", "")


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def emoji_for(category):
    return CLASS_EMOJI.get(category, "❓")


def photo(item):
    filename = item.get("filename", "")

    path = IMAGE_DIR / filename

    if filename and path.is_file():
        st.image(
            str(path),
            use_container_width=True
        )
    else:
        st.markdown(
            f"""
            <div style="
                font-size:3rem;
                text-align:center;
                padding:1rem;
            ">
                {emoji_for(item.get("category", ""))}
            </div>
            """,
            unsafe_allow_html=True
        )


def search_items(items, query):
    query = query.lower().strip()

    if not query:
        return items

    words = query.split()

    results = []

    for item in items:
        text = " ".join([
            item.get("name", ""),
            item.get("category", ""),
            item.get("description", ""),
            " ".join(item.get("search_terms", []))
        ]).lower()

        score = sum(
            1 for word in words
            if word in text
        )

        if score:
            results.append((score, item))

    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [item for _, item in results]


# ============================================================
# YOLO ERKENNUNG
# ============================================================

def detect_object(image):
    """
    Erkennt Gegenstände mit YOLO11n.
    Gibt das wahrscheinlichste relevante Objekt
    und die Top-Ergebnisse zurück.
    """

    results = model.predict(
        source=image,
        conf=0.15,
        iou=0.45,
        verbose=False
    )

    if not results:
        return None

    result = results[0]

    if result.boxes is None:
        return None

    detections = []

    for box in result.boxes:
        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        raw_name = result.names[class_id]

        raw_name = raw_name.lower()

        if raw_name not in YOLO_CATEGORIES:
            continue

        category = YOLO_CATEGORIES[raw_name]

        detections.append({
            "label": category,
            "confidence": confidence
        })

    if not detections:
        return None

    # Höchste Wahrscheinlichkeit zuerst
    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    best = detections[0]

    # Gleiche Kategorie nur einmal anzeigen
    unique = []

    seen = set()

    for detection in detections:
        label = detection["label"]

        if label not in seen:
            seen.add(label)
            uniqueamlit-Widgets für Upload/Suche.
""".append(detection)

    top3 = unique[:3]

    return {
        "label": best["label"],
        "category": best["label"],
        "confidence": best["confidence"],
        "top3": [
            (
                detection["label"],
                detection["confidence"]
            )
            for detection in top3
        ],
        "engine": "YOLO11n"
    }


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #FAFAF9;
    }

    .block-container {
        max-width: 950px !important;
        margin: auto;
    }

    .fund-title {
        font-weight: 800;
        letter-spacing: -0.03em;
        font-size: 3rem;
        color: #1C1917;
        margin-bottom: 0;
    }

    .fund-sub {
        color: #57534E;
        font-size: 1.05rem;
        margin-top: 0.2rem;
    }

    .hero {
        background: white;
        border: 1px solid #E7E5E4;
        border-left: 6px solid #B91C1C;
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    .category-card {
        background: white;
        border: 1px solid #E7E5E4;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .result-box {
        background: white;
        border: 1px solid #E7E5E4;
        border-radius: 15px;
        padding: 1.2rem;
        margin-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="fund-title">Fundbox 🔎</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="fund-sub">'
    'Das Fundbüro des Katharineums zu Lübeck — '
    'Foto hochladen und Gegenstand erkennen lassen.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.markdown(
    f"""
    **Katharineum zu Lübeck** ·
    **{len(CLASSES)} YOLO-Kategorien** ·
    **YOLO11n**
    """
)

st.divider()amlit-Widgets für Upload/Suche.
"""


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔎 Browse",
    "🔍 Search",
    "📷 Report found item"
])


# ============================================================
# BROWSE
# ============================================================

with tab1:

    st.markdown(
        """
        <div class="hero">
            <h2>Lost something? 👀</h2>
            <p>
            Browse through the found items or report a new item
            using a photo.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Found items",
            len(items)
        )

    with c2:
        st.metric(
            "AI categories",
            len(CLASSES)
        )

    st.subheader("Categories")

    cols = st.columns(4)

    for index, category in enumerate(CLASSES):

        count = len([
            item
            for item in items
            if item.get("category") == category
        ])

        with cols[index % 4]:

            st.markdown(
                f"""
                <div class="category-card">
                    <h4>
                        {emoji_for(category)}
                        {category}
                    </h4>
                    <p>{count} items</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.subheader("Recently found")

    if not items:

        st.info(
            "No found items have been added yet."
        )

    else:

        recent = items[:8]

        cols = st.columns(4)

        for index, item in enumerate(recent):

            with cols[index % 4]:

                photo(item)

                st.markdown(
                    f"**{item.get('name', 'Found item')}**"
                )

                st.caption(
                    item.get("category", "")
                )


# ============================================================
# SEARCH
# ============================================================

with tab2:

    st.subheader("Search")

    query = st.text_input(
        "Search",
        value=st.session_state.get("search", ""),
        placeholder="e.g. backpack, bottle, laptop..."
    )

    category = st.selectbox(
        "Category",
        ["All"] + CLASSES
    )

    results = search_items(
        items,
        query
    )

    if category != "All":

        results = [
            item
            for item in results
            if item.get("category") == category
        ]

    st.write(
        f"**{len(results)}** results"
    )

    if not results:

        st.info(
            "No matching found items."
        )

    else:

        for item in results:

            with st.container():

                col1, col2 = st.columns([1, 2])

                with col1:
                    photo(item)

                with col2:

                    st.subheader(
                        f"{emoji_for(item.get('category'))} "
                        f"{item.get('name', 'Found item')}"
                    )

                    st.write(
                        f"Category: "
                        f"**{item.get('category', '')}**"
                    )

                    if item.get("description"):
                        st.write(
                            item["description"]
                        )

                    if item.get("created_at"):
                        st.caption(
                            "Found on "
                            + item["created_at"][:10]
                        )

                    if st.button(
                        "Collected ✅",
                        key=f"remove_{item['id']}"
                    ):

                        items.remove(item)

                        save_items(items)

                        st.rerun()

                st.divider()


# ============================================================
# REPORT FOUND ITEM
# ============================================================

with tab3:

    st.subheader(
        "Report a found item 📷"
    )

    st.write(
        "Upload a photo and YOLO11n will try "
        "to identify the object."
    )

    source = st.radio(
        "Image source",
        [
            "Upload image",
            "Camera"
        ],
        horizontal=True
    )

    uploader_key = (
        f"uploader_{st.session_state['uploader_nonce']}"
    )

    if source == "Upload image":

        uploaded = st.file_uploader(
            "Choose an image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ],
            key=uploader_key
        )

    else:

        uploaded = st.camera_input(
            "Take a photo",
            key=uploader_key
        )

    if uploaded is not None:

        st.session_state["rep_bytes"] = (
            uploaded.getvalue()
        )

    if not st.session_state.get("rep_bytes"):

        st.info(
            "Upload or take a photo to start."
        )

    else:

        image = Image.open(
            io.BytesIO(
                st.session_state["rep_bytes"]
            )
        ).convert("RGB")

        col1, col2 = st.columns(
            [1, 1]
        )

        with col1:

            st.image(
                image,
                caption="Preview",
                use_container_width=True
            )

        with col2:

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            if st.button(
                "🔎 Detect object",
                use_container_width=True
            ):

                with st.spinner(
                    "YOLO11n is analyzing the image..."
                ):

                    detection = detect_object(
                        image
                    )

                st.session_state["rep_ai"] = (
                    detection
                )

                if detection:

                    st.session_state[
                        "rep_name_ai"
                    ] = detection["label"]

                else:

                    st.session_state[
                        "rep_name_ai"
                    ] = ""

                st.rerun()

            ai = st.session_state.get(
                "rep_ai"
            )

            if ai:

                st.success(
                    f"Detected: "
                    f"{emoji_for(ai['label'])} "
                    f"**{ai['label']}**"
                )

                st.progress(
                    min(
                        1.0,
                        max(
                            0.0,
                            ai["confidence"]
                        )
                    )
                )

                st.caption(
                    f"Confidence: "
                    f"{ai['confidence'] * 100:.1f}%"
                )

                if ai.get("top3"):

                    st.write(
                        "**Other detections:**"
                    )

                    for label, probability in ai["top3"]:

                        st.write(
                            f"{emoji_for(label)} "
                            f"{label} — "
                            f"{probability * 100:.1f}%"
                        )

            elif ai is not None:

                st.warning(
                    "YOLO11n could not identify "
                    "a relevant object in this photo."
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        st.divider()

        st.subheader(
            "Add to the lost & found"
        )

        ai = st.session_state.get(
            "rep_ai"
        ) or {}

        default_name = st.session_state.get(
            "rep_name_ai",
            ""
        )

        name = st.text_input(
            "Item name",
            value=default_name,
            placeholder="e.g. Black backpack"
        )

        detected_category = ai.get(
            "category"
        )

        if (
            detected_category
            and detected_category in CLASSES
        ):

            default_index = CLASSES.index(
                detected_category
            )

        else:

            default_index = 0

        category = st.selectbox(
            "Category",
            CLASSES,
            index=default_index
        )

        description = st.text_area(
            "Description",
            placeholder=(
                "Color, brand, special details..."
            )
        )

        if st.button(
            "Put item in Fundbox ✅",
            type="primary",
            use_container_width=True
        ):

            if not name.strip():

                st.error(
                    "Please enter an item name."
                )

            else:

                item_id = str(
                    uuid.uuid4()
                )

                filename = (
                    item_id + ".jpg"
                )

                image.save(
                    IMAGE_DIR / filename,
                    format="JPEG",
                    quality=85
                )

                new_item = {
                    "id": item_id,
                    "name": name.strip(),
                    "category": category,
                    "description": (
                        description.strip()
                        or f"Found item: {name.strip()}."
                    ),
                    "search_terms": [
                        name.lower(),
                        category.lower()
                    ],
                    "filename": filename,
                    "created_at": (
                        datetime.datetime.now()
                        .isoformat(
                            timespec="seconds"
                        )
                    )
                }

                items.insert(
                    0,
                    new_item
                )

                save_items(items)

                st.session_state[
                    "rep_bytes"
                ] = None

                st.session_stateamlit-Widgets für Upload/Suche.
""" = None

                st.session_state[
                    "rep_name_ai"
                ] = ""

                st.session_state[
                    "uploader_nonce"
                ] += 1

                st.success(
                    f"'{name.strip()}' was added "
                    "to the Fundbox! 🎉"
                )

                st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Fundbox · Katharineum zu Lübeck · "
    "YOLO11n object detection 🔎"
)
```
