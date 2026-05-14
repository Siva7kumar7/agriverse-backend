import os
import sys
import logging
import cv2
import numpy as np
import tensorflow as tf
import joblib
import json
import base64
from flask_bcrypt import Bcrypt
from flask import Flask, request, jsonify
from flask_cors import CORS
from bson import ObjectId
from weather.routes import weather_bp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from db import user_collection, product_collection, cart_collection, order_collection

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================================================
# LOGGING SETUP
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
bcrypt = Bcrypt(app)

app.register_blueprint(weather_bp, url_prefix="/api")


# ================= REGISTER =================
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role")

    # Check existing user
    if user_collection.find_one({"email": email}):
        return jsonify({"message": "User already exists"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    user_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password,
        "role": role
    })

    return jsonify({"message": "Registration successful"})


# ================= LOGIN =================
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json

    email = data.get("email")
    password = data.get("password")
    role = data.get("role")

    user = user_collection.find_one({"email": email, "role": role})

    if user and bcrypt.check_password_hash(user["password"], password):
        return jsonify({
            "message": "Login successful",
            "name": user["name"],
            "role": user["role"],
            "email": user["email"],
            "token": "agriverse-session-active"
        })
    else:
        return jsonify({"message": "Invalid credentials"}), 401


# ================= FORGOT PASSWORD =================
@app.route("/api/forgot-password", methods=["POST"])
def forgot_password():
    try:
        data = request.json
        email = data.get("email")
        new_password = data.get("newPassword")

        user = user_collection.find_one({"email": email})

        if not user:
            return jsonify({"message": "User not found"}), 404

        hashed_password = bcrypt.generate_password_hash(new_password).decode("utf-8")

        user_collection.update_one(
            {"email": email},
            {"$set": {"password": hashed_password}}
        )

        return jsonify({"message": "Password updated successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= PLACE ORDER =================
@app.route("/api/orders", methods=["POST"])
def place_order():
    data = request.json

    order = {
        "userId": data.get("userId"),
        "items": data.get("items"),
        "total": data.get("total"),
        "address": data.get("address"),
        "payment": data.get("payment"),
        "status": "Pending"
    }

    order_collection.insert_one(order)

    # Clear cart after order
    cart_collection.delete_many({"userId": data.get("userId")})

    return jsonify({"message": "Order placed successfully"})


# ================= ADD ORDER =================
@app.route("/api/orders/add", methods=["POST"])
def add_order():
    try:
        data = request.json

        order = {
            "userId": data.get("userId"),
            "items": data.get("items"),
            "total": data.get("total"),
            "address": data.get("address"),
            "payment": data.get("payment"),
            "status": "Placed"
        }

        order_collection.insert_one(order)

        return jsonify({"message": "Order placed successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= GET ORDERS =================
@app.route("/api/orders", methods=["GET"])
def get_orders():
    orders = list(order_collection.find())
    for o in orders:
        o["_id"] = str(o["_id"])
    return jsonify(orders)


# ================= GET USER ORDERS =================
@app.route("/api/orders/user/<user_id>", methods=["GET"])
def get_user_orders(user_id):
    try:
        orders = list(order_collection.find({"userId": user_id}))
        for o in orders:
            o["_id"] = str(o["_id"])
        return jsonify(orders)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= UPDATE ORDER =================
@app.route("/api/orders/<id>", methods=["PUT"])
def update_order(id):
    data = request.json
    order_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"status": data.get("status")}}
    )
    return jsonify({"message": "Updated"})


# ================= GET CART =================
@app.route("/api/cart/<user_id>", methods=["GET"])
def get_cart(user_id):
    try:
        cart_items = list(cart_collection.find({"userId": user_id}))
        for item in cart_items:
            item["_id"] = str(item["_id"])
        return jsonify({"items": cart_items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= ADD TO CART =================
@app.route("/api/cart/add", methods=["POST"])
def add_to_cart():
    try:
        data = request.json
        user_id = data.get("userId")
        product_id = data.get("productId")

        # Check if item already in cart
        existing = cart_collection.find_one({"userId": user_id, "productId": product_id})
        if existing:
            cart_collection.update_one(
                {"_id": existing["_id"]},
                {"$inc": {"quantity": 1}}
            )
        else:
            item = {
                "userId": user_id,
                "productId": product_id,
                "name": data.get("name"),
                "price": data.get("price"),
                "image": data.get("image"),
                "quantity": 1
            }
            cart_collection.insert_one(item)

        return jsonify({"message": "Added to cart"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= REMOVE FROM CART =================
@app.route("/api/cart/remove/<user_id>/<product_id>", methods=["DELETE"])
def remove_from_cart(user_id, product_id):
    try:
        cart_collection.delete_one({"userId": user_id, "productId": product_id})
        return jsonify({"message": "Removed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= UPDATE CART QTY =================
@app.route("/api/cart/update", methods=["PUT"])
def update_cart_qty():
    try:
        data = request.json
        user_id = data.get("userId")
        product_id = data.get("productId")
        action = data.get("action")

        inc = 1 if action == "increase" else -1
        
        cart_collection.update_one(
            {"userId": user_id, "productId": product_id},
            {"$inc": {"quantity": inc}}
        )
        
        # Remove if qty becomes 0
        cart_collection.delete_many({"userId": user_id, "quantity": {"$lte": 0}})

        return jsonify({"message": "Updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= ADD PRODUCT (with image upload) =================
@app.route("/api/products/add", methods=["POST"])
def add_product():
    try:
        name = request.form.get("name")
        price = request.form.get("price")
        category = request.form.get("category")
        location = request.form.get("location")
        farmer = request.form.get("farmer", "Farmer")

        image_file = request.files.get("image")

        image_data = None

        if image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        product = {
            "name": name,
            "price": float(price),
            "category": category,
            "location": location,
            "farmer": farmer,
            "image": image_data
        }

        product_collection.insert_one(product)

        return jsonify({"message": "Product added successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================= GET PRODUCTS =================
@app.route("/api/products", methods=["GET"])
def get_products():
    product_list = []

    for p in product_collection.find():
        p["_id"] = str(p["_id"])
        if "farmerId" in p:
            p["farmerId"] = str(p["farmerId"])
        product_list.append(p)

    return jsonify(product_list)


# ================= DELETE PRODUCT =================
@app.route("/api/products/<id>", methods=["DELETE"])
def delete_product(id):
    product_collection.delete_one({"_id": ObjectId(id)})
    return jsonify({"message": "Deleted"})


# ================= UPDATE PRODUCT =================
@app.route("/api/products/update/<id>", methods=["PUT"])
def update_product(id):
    data = request.json

    product_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {
            "price": data["price"],
            "quantity": data["quantity"],
            "category": data["category"],
            "location": data["location"]
        }}
    )

    return jsonify({"message": "Product updated successfully"})


# =====================================================
# PLANT DISEASE MODEL PATHS
# =====================================================
MODEL_DIR = os.path.join(BASE_DIR, "plant_disease")

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_classifier.h5")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "svm_scaler.pkl")
SEVERITY_MODEL_PATH = os.path.join(MODEL_DIR, "severity_regressor.pkl")

AGRI_KNOWLEDGE_PATH = os.path.join(MODEL_DIR, "agri_knowledge.json")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")

IMG_SIZE = (224, 224)
CONF_THRESHOLD = 0.5

# Models will be loaded lazily to prevent startup crashes
cnn_model = None
svm = None
scaler = None
severity_model = None
feature_extractor = None

def load_plant_models():
    global cnn_model, svm, scaler, severity_model, feature_extractor
    if cnn_model is not None:
        return True
    
    try:
        logger.info("🌿 Loading plant disease models lazily...")
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
        svm = joblib.load(SVM_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        if os.path.exists(SEVERITY_MODEL_PATH):
            severity_model = joblib.load(SEVERITY_MODEL_PATH)
            
        feature_extractor = tf.keras.Model(
            inputs=cnn_model.input,
            outputs=cnn_model.get_layer("feature_layer").output
        )
        logger.info("✅ Plant disease models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load plant models: {e}")
        return False

# =====================================================
# LOAD JSON DATA
# =====================================================
with open(AGRI_KNOWLEDGE_PATH) as f:
    agri_knowledge = json.load(f)

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}

# =====================================================
# UTIL FUNCTIONS
# =====================================================
def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def severity_label(percent):
    if percent >= 80:
        return "High"
    elif percent >= 60:
        return "Moderate"
    return "Low"

# =====================================================
# HEALTH CHECK
# =====================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "OK",
        "message": "Backend running successfully"
    })

# =====================================================
# PLANT DISEASE DETECTION API
# =====================================================
@app.route("/api/plant/detect", methods=["POST"])
def detect_plant_disease():
    try:
        if not load_plant_models():
            return jsonify({"error": "AI Model service is temporarily unavailable"}), 503

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        img_input = preprocess_image(img)

        # Feature extraction
        features = feature_extractor.predict(img_input, verbose=0)
        features_scaled = scaler.transform(features)

        # SVM prediction
        probs = svm.predict_proba(features_scaled)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        if confidence < CONF_THRESHOLD:
            return jsonify({"error": "Leaf not detected clearly"}), 400

        disease = index_to_class[class_id]
        remedy = agri_knowledge[disease]["remedy"]
        fertilizer = agri_knowledge[disease]["fertilizer"]

        severity_percent = confidence * 100
        if severity_model:
            severity_percent = float(severity_model.predict(features)[0])

        response = {
            "disease": disease.replace("___", " - "),
            "severity": round(severity_percent, 1),
            "severity_level": severity_label(severity_percent),
            "fertilizer": fertilizer,
            "remedy": remedy,
            "confidence": round(confidence * 100, 2)
        }

        return jsonify(response)

    except Exception as e:
        logger.exception("Plant disease detection error")
        return jsonify({"error": str(e)}), 500

# =====================================================
# START SERVER
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask Server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
