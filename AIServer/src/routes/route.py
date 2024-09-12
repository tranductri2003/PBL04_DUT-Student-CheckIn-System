from flask import Blueprint
from flask import request, jsonify
import jwt

from actions.face_recognization import create_features, face_recognize, read_image
from settings import JWT_SECRET_KEY

api_v1 = Blueprint('api_v1', __name__)

@api_v1.route('/', methods=["GET"])
def home():
    return jsonify({"msg":"Hello"}), 200

@api_v1.route('/create-image-features', methods=["POST"])
def add_image_feature():
    decoded_token = jwt.decode(jwt=request.headers.get('Authorization').split(' ')[1], key=JWT_SECRET_KEY, algorithms=['HS256'])
    if decoded_token.get('role', None) == None:
        return jsonify({"msg": "Unauthorized"}), 401
    images = request.files.getlist('image')  # Sử dụng request.files.getlist thay vì request.files['image']
    staff_id = decoded_token.get('staff_id')
    try:
        create_features(staff_id, images)
        return jsonify({"msg": "Create successfully!"}), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"msg": "Create failed!"}), 400
    

@api_v1.route('/face-recognization', methods=["POST"])
def face_recognization():
    try:
        decoded_token = jwt.decode(jwt=request.headers.get('Authorization').split(' ')[1], key=JWT_SECRET_KEY, algorithms=['HS256'])
        if decoded_token.get('role', None) == None:
            return jsonify({"msg": "Unauthorized"}), 401
        staff_id = decoded_token.get('staff_id')
        image = request.files['image']
        
        # Đọc hình ảnh từ đối tượng FileStorage
        validated = face_recognize(staff_id, image, 0.5)
        return jsonify({"validated": validated}), 200
    except Exception as e: 
        print("Error:", str(e))
        return jsonify({"msg": "Recognize failed!"}), 400