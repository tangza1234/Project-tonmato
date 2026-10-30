import os
import threading

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError

# ใช้ตัวรัน TFLite แบบเบา แทน TensorFlow เต็มตัว (กินแรมน้อยกว่ามาก)
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # เผื่อรันในเครื่องที่มีแต่ tensorflow
    from tensorflow.lite.python.interpreter import Interpreter

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024  # จำกัดไฟล์ไม่เกิน ~6MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.tflite')

IMG_SIZE = (224, 224)

# ต้องตรงกับตอนเทรนในโน้ตบุ๊ก (USE_CENTER_CROP / CENTER_CROP_RATIO)
USE_CENTER_CROP = True
CENTER_CROP_RATIO = 0.6

# ---------- เกณฑ์ตัดสินใจ (ปรับได้) ----------
CONF_HIGH = 70.0              # % ตั้งแต่ค่านี้ขึ้นไป = ตอบมั่นใจ
CONF_MIN = 50.0               # % ต่ำกว่านี้ = "ไม่สามารถระบุได้"; ระหว่าง CONF_MIN-CONF_HIGH = ตอบพร้อมคำเตือน "ความมั่นใจปานกลาง"
MIN_LEAF_RATIO = 0.15         # สัดส่วนพิกเซล "สีใบพืช" ขั้นต่ำในส่วนกึ่งกลางภาพ
NOT_TOMATO_MIN_CONF = 50.0    # โมเดล 6 คลาส: ทาย Not_Tomato_Leaf ที่ความมั่นใจเท่าไรถึงจะเชื่อ
# โมเดลบอก "ไม่ใช่ใบมะเขือเทศ" แต่ภาพมีสีพืชมากกว่าค่านี้ (สัดส่วนพิกเซล) = อาจเป็นมะเขือเทศที่ถ่ายกว้าง/ไกล/ทั้งต้น
# โมเดลมักมั่นใจเกินไปกับภาพแบบนี้ จึงไม่ฟันธง แต่ตอบข้อความกลาง ๆ ให้ถ่ายใหม่แทน
NOT_TOMATO_SOFT_LEAF_RATIO = 0.30

# ลำดับคลาสต้องตรงกับ train_data.class_indices ของโน้ตบุ๊ก
# (flow_from_directory เรียงตามตัวอักษร: "Not_Tomato_Leaf" มาก่อน "Tomato___...")
NOT_TOMATO = "Not_Tomato_Leaf"
CLASSES_5 = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]
CLASSES_6 = [NOT_TOMATO] + CLASSES_5

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

# ---------- โหลดโมเดล TFLite ----------
interpreter = Interpreter(model_path=MODEL_PATH, num_threads=1)
interpreter.allocate_tensors()
_input = interpreter.get_input_details()[0]
_output = interpreter.get_output_details()[0]
_lock = threading.Lock()  # Interpreter ไม่ thread-safe

_n_out = int(_output['shape'][-1])
if _n_out == len(CLASSES_6):
    CLASSES = CLASSES_6      # โมเดลที่มีคลาส Not_Tomato_Leaf
elif _n_out == len(CLASSES_5):
    CLASSES = CLASSES_5      # โมเดล 5 คลาส ใช้ตัวกรองสีใบพืชช่วยแทน
else:
    raise RuntimeError(
        f"โมเดลมี {_n_out} เอาต์พุต แต่แอปรองรับ 5 หรือ 6 คลาสเท่านั้น — "
        f"ตรวจสอบว่าใช้ไฟล์โมเดลที่เทรนจากโน้ตบุ๊กถูกตัว"
    )
HAS_NOT_TOMATO_CLASS = NOT_TOMATO in CLASSES


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(file_stream):
    """ทำให้เหมือนตอนเทรนในโน้ตบุ๊ก:
    1) ย่อภาพเป็น 224x224 แบบ nearest (ค่าเริ่มต้นของ keras load_img)
    2) ครอปกึ่งกลาง CENTER_CROP_RATIO แล้วขยายกลับเป็น 224x224 แบบ bilinear
    3) หารด้วย 255
    """
    img = Image.open(file_stream).convert('RGB').resize(IMG_SIZE, Image.NEAREST)

    if USE_CENTER_CROP:
        w, h = img.size
        crop_h = int(h * CENTER_CROP_RATIO)
        crop_w = int(w * CENTER_CROP_RATIO)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        img = img.crop((left, top, left + crop_w, top + crop_h)).resize(IMG_SIZE, Image.BILINEAR)

    return img


def leaf_ratio(img):
    """สัดส่วนพิกเซลที่เป็น "สีพืช" (เหลืองอมเขียว/เขียว) ในภาพที่โมเดลมองจริง
    ใช้กรองรูปที่ไม่ใช่ใบพืชเลย เช่น ภาพหน้าจอ เอกสาร ท้องฟ้า วัตถุสีล้วน สีผิว
    (ใบที่เป็นโรคยังมีส่วนสีเขียว/เหลืองปนอยู่ จึงยังผ่านเกณฑ์)"""
    small = img.resize((112, 112), Image.BOX)          # เฉลี่ยสีก่อน ลด noise
    hsv = np.asarray(small.convert('HSV'), dtype=np.float32)
    hue = hsv[..., 0] * 360.0 / 255.0
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    veg = (hue >= 35) & (hue <= 165) & (sat >= 0.25) & (val >= 0.20)
    return float(veg.mean())


def to_batch(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def run_model(batch):
    with _lock:
        interpreter.set_tensor(_input['index'], batch)
        interpreter.invoke()
        return interpreter.get_tensor(_output['index'])[0].copy()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/healthz')
def healthz():
    return 'ok'


@app.errorhandler(413)
def too_large(_e):
    return jsonify({'error': 'ไฟล์ใหญ่เกิน 5MB'}), 413


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'กรุณาอัปโหลดรูปภาพ'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'กรุณาเลือกรูปภาพ'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'รองรับเฉพาะไฟล์ PNG, JPG, JPEG, WEBP เท่านั้น'}), 400

    try:
        img = preprocess(file.stream)
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({'error': 'ไฟล์ภาพเสียหายหรืออ่านไม่ได้'}), 400

    probs = run_model(to_batch(img))

    class_index = int(np.argmax(probs))
    ai_confidence = round(float(probs[class_index]) * 100, 2)
    model_result = CLASSES[class_index]

    ratio = leaf_ratio(img)

    top_idx = np.argsort(probs)[-3:][::-1]
    top3 = [
        {
            "class": CLASSES[i],
            "name": DISEASE_INFO.get(CLASSES[i], {}).get("name_th", CLASSES[i]),
            "confidence": round(float(probs[i]) * 100, 2)
        }
        for i in top_idx
    ]

    disease_info = None
    soft_not_tomato = False
    if model_result == NOT_TOMATO and ai_confidence >= NOT_TOMATO_MIN_CONF:
        if ratio >= NOT_TOMATO_SOFT_LEAF_RATIO:
            # ภาพมีพืชเขียวเยอะ: อาจเป็นใบมะเขือเทศที่ถ่ายกว้าง/ไกล/ทั้งต้น หรือพืชอื่น ไม่ฟันธง
            result_text, result_type = (
                "ไม่พบใบมะเขือเทศที่ชัดเจนในภาพ — ภาพอาจกว้างหรือไกลเกินไป หรือเป็นพืชชนิดอื่น "
                "ลองถ่ายใบมะเขือเทศ 1 ใบ ระยะใกล้ให้เต็มภาพ"
            ), "unknown"
            soft_not_tomato = True
        else:
            # โมเดลเองบอกว่าไม่ใช่ใบมะเขือเทศ และภาพมีสีพืชน้อย
            result_text, result_type = "ไม่ใช่ใบมะเขือเทศ", "not_tomato"
    elif ratio < MIN_LEAF_RATIO:
        # ภาพแทบไม่มีสีพืชเลย ไม่น่าใช่ใบ
        result_text, result_type = "ไม่ใช่ใบมะเขือเทศ (ไม่พบใบพืชในภาพ)", "not_tomato"
    elif ai_confidence < CONF_MIN or model_result == NOT_TOMATO:
        result_text, result_type = "ไม่สามารถระบุได้ (ความมั่นใจต่ำ)", "unknown"
    else:
        info = DISEASE_INFO.get(model_result, {})
        result_text = info.get("name_th", model_result)
        result_type = "healthy" if model_result == "Tomato___healthy" else "disease"
        disease_info = info
        if ai_confidence < CONF_HIGH:
            result_text += " (ความมั่นใจปานกลาง ควรตรวจซ้ำ)"

    return jsonify({
        'result': result_text,
        'result_type': result_type,
        # กรณีไม่ฟันธง ไม่แสดงตัวเลข/อันดับของโมเดล เพราะ "Not Tomato 99%" จะขัดกับข้อความที่บอกว่าไม่แน่ใจ
        'confidence': 0 if soft_not_tomato else ai_confidence,
        'hide_confidence': soft_not_tomato,
        'image_path': None,
        'disease_info': disease_info,
        'top3': [] if soft_not_tomato else top3
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
