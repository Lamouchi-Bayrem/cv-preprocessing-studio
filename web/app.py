from flask import Flask, request, send_file
import cv2
import numpy as np
import json
from core.pipeline import AdvancedCVPipeline

app = Flask(__name__)


@app.route("/process", methods=["POST"])
def process():

    file = request.files["image"]
    params = request.form.get("params")

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR,
    )

    config = json.loads(params)
    pipeline = AdvancedCVPipeline(config)

    normalized, output = pipeline.preprocess(img)

    cv2.imwrite("output.png", output)
    return send_file("output.png", mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
