import csv
import os
import random

from flask import Flask, render_template, request, redirect

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model_1_latest.pt")


def load_csv(name):
    with open(os.path.join(BASE_DIR, name), encoding="cp1252", newline="") as f:
        return list(csv.DictReader(f))


disease_info = load_csv("disease_info.csv")
supplement_info = load_csv("supplement_info.csv")

# ---------------------------------------------------------------------------
# Model loading.
# The real CNN needs PyTorch + the trained .pt file (~200 MB). These are too big
# for Vercel's serverless limit, so when either is missing the app runs in
# DEMO MODE: the full UI works and the result is picked from the file name.
# ---------------------------------------------------------------------------
model = None
try:
    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    import CNN

    if os.path.exists(MODEL_PATH):
        model = CNN.CNN(39)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
except ImportError:
    model = None

DEMO_MODE = model is None


def model_predict(file_storage):
    image = Image.open(file_storage.stream).convert("RGB").resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    with torch.no_grad():
        output = model(input_data).numpy()
    return int(np.argmax(output))


def demo_predict(filename):
    """Pick the class whose name best matches the uploaded file name."""
    base = os.path.splitext(filename.lower())[0]
    for ch in "-. ,()":
        base = base.replace(ch, "_")
    words = [w for w in base.split("_") if len(w) > 2]
    best, best_score = None, 0
    for i, row in enumerate(disease_info):
        name = row["disease_name"].lower()
        score = sum(w in name for w in words)
        if score > best_score:
            best, best_score = i, score
    return best if best is not None else random.randrange(len(disease_info))


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # Vercel request body limit is ~4.5 MB


@app.context_processor
def inject_mode():
    return {"demo_mode": DEMO_MODE}


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/contact")
def contact():
    return render_template("contact-us.html")


@app.route("/index")
def ai_engine_page():
    return render_template("index.html")


@app.route("/submit", methods=["GET", "POST"])
def submit():
    image = request.files.get("image")
    if request.method != "POST" or image is None or image.filename == "":
        return redirect("/index")

    pred = demo_predict(image.filename) if DEMO_MODE else model_predict(image)

    d, s = disease_info[pred], supplement_info[pred]
    return render_template(
        "submit.html",
        title=d["disease_name"],
        desc=d["description"],
        prevent=d["Possible Steps"],
        image_url=d["image_url"],
        pred=pred,
        sname=s["supplement name"],
        simage=s["supplement image"],
        buy_link=s["buy link"],
    )


@app.route("/market")
def market():
    return render_template(
        "market.html",
        supplement_image=[r["supplement image"] for r in supplement_info],
        supplement_name=[r["supplement name"] for r in supplement_info],
        disease=[r["disease_name"] for r in disease_info],
        buy=[r["buy link"] for r in supplement_info],
    )


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="Image is too large. Please upload an image under 4 MB."), 413


if __name__ == "__main__":
    print("Mode:", "DEMO (no model/torch found)" if DEMO_MODE else "REAL AI MODEL")
    app.run(debug=True)
