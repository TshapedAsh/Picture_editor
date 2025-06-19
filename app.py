from flask import Flask, request, jsonify, send_file
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2
import numpy as np
import tempfile
from PIL import Image

app = Flask(__name__)

# Initialize face detector and swapper
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)
swapper = get_model("inswapper_128.onnx")

@app.route('/swap', methods=['POST'])
def swap_faces():
    source = request.files['source']
    target = request.files['target']
    
    src_img = np.array(Image.open(source).convert("RGB"))
    tgt_img = np.array(Image.open(target).convert("RGB"))

    src_faces = face_app.get(src_img)
    tgt_faces = face_app.get(tgt_img)
    if not src_faces or not tgt_faces:
        return jsonify({"error": "No face detected!"}), 400

    result = swapper.get(tgt_img, tgt_faces[0], src_faces[0])

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp.name, result)
    return send_file(temp.name, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run()
