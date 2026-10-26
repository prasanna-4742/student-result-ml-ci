from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = Flask(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    "ml_http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "ml_http_request_duration_seconds",
    "HTTP Request Latency",
    ["endpoint"]
)

PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total Predictions",
    ["prediction"]
)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found. Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


# Root Endpoint
@app.get("/")
def root_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


# Health Endpoint
@app.route("/health")
def health():
    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/health"
    ).inc()

    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    })


# Prediction Endpoint
@app.post("/predict")
def predict():

    start_time = time.time()

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict"
    ).inc()

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        f for f in FEATURES if f not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([
        {f: data[f] for f in FEATURES}
    ])

    model = load_model()

    prediction_code = int(
        model.predict(sample)[0]
    )

    prediction = (
        "PASS"
        if prediction_code == 1
        else "FAIL"
    )

    REQUEST_LATENCY.labels(
        endpoint="/predict"
    ).observe(
        time.time() - start_time
    )

    PREDICTION_COUNT.labels(
        prediction=prediction
    ).inc()

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code
    })


# Prometheus Metrics Endpoint
@app.route("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
