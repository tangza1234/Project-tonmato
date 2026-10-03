import os
import uuid
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

MODEL_PATH = 'final_model.h5'
model = load_model(MODEL_PATH)

classes = [
    "Not_Tomato_Leaf",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

DISEASE_INFO = {
    "Tomato___Bacterial_spot": {
        "name_th": "Tomato Bacterial Spot",
        "cause": "แบคทีเรีย Xanthomonas",
        "symptom": "จุดสีน้ำตาลดำขนาดเล็ก มีขอบสีเหลืองรอบจุด",
        "treatment": "ใช้สารทองแดง (Copper-based) พ่น, หลีกเลี่ยงการให้น้ำแบบ overhead"
    },
    "Tomato___Early_blight": {
        "name_th": "Tomato Early Blight",
        "cause": "เชื้อรา Alternaria linariae",
        "symptom": "จุดสีน้ำตาลมีวงซ้อนกันคล้ายเป้า มักเริ่มที่ใบล่างก่อน",
        "treatment": "ตัดใบที่เป็นโรคออก, พ่นสารป้องกันเชื้อรากลุ่ม Chlorothalonil"
    },
    "Tomato___Late_blight": {
        "name_th": "Tomato Late Blight",
        "cause": "เชื้อรา Phytophthora infestans",
        "symptom": "แผลสีน้ำตาลเข้มชุ่มน้ำ มีเส้นใยขาวใต้ใบ ลามเร็วมาก",
        "treatment": "พ่นสาร Metalaxyl หรือ Mancozeb, ทำลายต้นที่เป็นโรคหนักทันที"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name_th": "Tomato Yellow Leaf Curl Virus",
        "cause": "ไวรัส TYLCV แพร่โดยแมลงหวี่ขาว (Whitefly)",
        "symptom": "ใบม้วนงอ สีเหลือง ต้นแคระแกร็น ไม่ออกผล",
        "treatment": "ไม่มียารักษาโดยตรง ควบคุมแมลงหวี่ขาว, ถอนต้นที่เป็นโรคออก"
    },
    "Tomato___healthy": {
        "name_th": "Tomato Healthy",
        "cause": "-",
        "symptom": "ใบสีเขียวสด ไม่มีจุดหรือรอยโรค",
        "treatment": "ดูแลตามปกติ รดน้ำและใส่ปุ๋ยอย่างเหมาะสม"
    }
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_unique_filename(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'กรุณาอัปโหลดรูปภาพ'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'กรุณาเลือกรูปภาพ'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'รองรับเฉพาะไฟล์ PNG, JPG, JPEG, WEBP เท่านั้น'}), 400

    unique_filename = get_unique_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array, verbose=0)
    class_index = np.argmax(predictions[0])
    ai_confidence = round(float(predictions[0][class_index]) * 100, 2)
    model_result = classes[class_index]

    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    top3 = [
        {
            "class": classes[idx],
            "name": DISEASE_INFO.get(classes[idx], {}).get("name_th", classes[idx]),
            "confidence": round(float(predictions[0][idx]) * 100, 2)
        }
        for idx in top3_idx
    ]

    if ai_confidence < 70:
        result_text = "ไม่สามารถระบุได้ (ความมั่นใจต่ำ)"
        result_type = "unknown"
        disease_info = None
    elif class_index == 0:
        result_text = "ไม่ใช่ใบมะเขือเทศ"
        result_type = "not_tomato"
        disease_info = None
    else:
        info = DISEASE_INFO.get(model_result, {})
        result_text = info.get("name_th", model_result)
        result_type = "healthy" if model_result == "Tomato___healthy" else "disease"
        disease_info = info

    return jsonify({
        'result': result_text,
        'result_type': result_type,
        'confidence': ai_confidence,
        'image_path': '/' + filepath.replace('\\', '/'),
        'disease_info': disease_info,
        'top3': top3
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
