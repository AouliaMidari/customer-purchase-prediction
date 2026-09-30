import os
import json
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("meta.json", "r") as f:
    meta = json.load(f)

FEATURES = meta["features_order"]
NUMERIC = list(meta["numeric"].keys())
CATEGORICAL = list(meta["categorical"].keys())
TARGET_LABELS = meta["target_labels"]


@app.route("/")
def home():
    return render_template(
        "index.html",
        numeric_fields=NUMERIC,
        categorical_fields=CATEGORICAL,
        categories=meta["categorical"],
        ranges=meta["numeric"],
    )


@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        row = {col: request.form[col] for col in FEATURES}
        X_new = pd.DataFrame([row], columns=FEATURES)
        for col in NUMERIC:
            X_new[col] = X_new[col].astype(float)

        prediction = int(model.predict(X_new)[0])
        label = TARGET_LABELS[str(prediction)]
        return render_template("result.html", result={"Prédiction": label})

    return render_template(
        "index.html",
        numeric_fields=NUMERIC,
        categorical_fields=CATEGORICAL,
        categories=meta["categorical"],
        ranges=meta["numeric"],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)