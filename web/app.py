from flask import Flask, request, send_file
import cv2
import numpy as np
from core.pipeline import PreprocessingPipeline
from core.config import ProcessingConfig

app = Flask(__name__)
pipeline = PreprocessingPipeline(ProcessingConfig())

@app.route("/process", methods=["POST"])
def process():
    file = request.files["image"]
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR,
    )

    binary, _ = pipeline.run(img)
    cv2.imwrite("out.png", binary)
    return send_file("out.png", mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
