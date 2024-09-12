from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import jwt
from settings import JWT_SECRET_KEY
from flask import request, jsonify
import requests
import io
import math
from sklearn.cluster import KMeans
from collections import defaultdict

import os
from db import *

face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

# Tải mô hình VGG16 (chọn include_top=False để loại bỏ các lớp fully connected ở cuối)
base_model = VGG16(weights='imagenet', include_top=False)

def read_image(image):
    image_stream = image.read()
    img_array = np.frombuffer(image_stream, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (1024, 1024))
    return img

def detect_face(image):
    print(face_cascade.__str__())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        # No faces detected
        return None

    # Return only the first detected face
    x, y, w, h = faces[0]
    face_roi = image[y:y+h, x:x+w]

    # Resize the face region to (224, 224, 3)
    face_resized = cv2.resize(face_roi, (224, 224))

    return face_resized

def extract_features(image):
    image_array = read_image(image)
    print(type(image))
    face = detect_face(image_array)
    if face is None:
        print("No face detected")
        return None
    print(type(image))
    x = np.expand_dims(face, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    features_vector = features.flatten()
    return features_vector


def normalize_vector(vector):
    max_value = np.max(np.abs(vector))
    if max_value > 0:
        normalized_vector = vector / max_value
    else:
        normalized_vector = vector
    return normalized_vector


def euclidean_distance(vector1, vector2):
    distance = 0
    for i in range(len(vector1)):
        distance += (vector1[i] - vector2[i])**2
    return math.sqrt(distance)

def face_recognize(student_id, image, threshold):
    check_vector = extract_features(image)
    if check_vector is None:
        return False
    
    check_vector = normalize_vector(check_vector)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute('SELECT staff_id, features FROM FaceFeatures')  # Chọn cột staff_id và features
    result = cur.fetchall()  # Lấy tất cả các dòng kết quả
    cur.close()
    close_connection(conn)

    # Duyệt qua từng dòng kết quả và lấy ra staff_id và features_vector tương ứng
    student_vector = defaultdict(lambda: 0)
    min_distance = 10**9
    for row in result:
        staff_id, features_vector = row
        vector1, vector2 = features_vector.split('#')
        
        vector1 = list(vector1.split())
        for i in range(len(vector1)):
            vector1[i] = float(vector1[i])
            
        vector2 = list(vector2.split())
        for i in range(len(vector2)):
            vector2[i] = float(vector2[i])
        
        
        if staff_id == student_id:
            correct_vector1 = np.array(vector1, dtype=float)
            correct_vector2 = np.array(vector2, dtype=float)

            # Calculate the mean of the vectors
            correct_vector = np.mean([correct_vector1, correct_vector2], axis=0)
        
        vector1 = normalize_vector(vector1)        
        vector2 = normalize_vector(vector2)
        
        distance = min(euclidean_distance(check_vector, vector1), euclidean_distance(check_vector, vector2))
        min_distance = min(min_distance, distance)
        
        student_vector[distance] = staff_id

    # print("Sinh viên đăng nhập hệ thống", student_id)
    print("Sinh viên nhận diện được", student_vector[min_distance])
    print("Khoảng cách chi tiết", student_vector)
    
    #Tính toán độ tương đồng cosine giữa hai vectơ đặc trưng
    matrix = np.vstack([correct_vector, check_vector])
    cosine_sim = cosine_similarity(matrix)

    # Chuyển giá trị float32 thành float
    cosine_similarity_value = float(cosine_sim[0, 1])
    print("Tương đồng cosine giữa ảnh trong db và ảnh thực tế của người đó", cosine_similarity_value)
    
    return cosine_similarity_value >= threshold or student_vector[min_distance] == student_id

def create_features(staff_id, images):
    data_features_vectors = []
    for image in images:
        feature_vector = extract_features(image)
        if feature_vector is not None:
            data_features_vectors.append(feature_vector)
    
            
    data_features_vectors = np.array(data_features_vectors)
    normalized_array = data_features_vectors / np.max(np.abs(data_features_vectors))
    flattened_array = normalized_array.reshape(normalized_array.shape[0], -1)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(flattened_array)
    centroids = kmeans.cluster_centers_
    
    try:
        string_features_vector = []
        for centroid in centroids:
            temp_array = []
            for element in centroid:
                temp_array.append(str(element))
            temp_string = " ".join(temp_array)
            string_features_vector.append(temp_string)
        string_features_vector ="#".join(string_features_vector) 
    
        # Insert đặc trưng vào cơ sở dữ liệu
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM FaceFeatures WHERE staff_id = %s", (staff_id,))
        existing_record = cur.fetchone()


        if existing_record:
            # print("Existing")
            cur.execute("UPDATE FaceFeatures SET features = %s WHERE staff_id = %s", (string_features_vector, staff_id))
        else:
            # print("No existing features")
            cur.execute("INSERT INTO FaceFeatures (features, staff_id) VALUES (%s , %s)", (string_features_vector, staff_id))

        conn.commit()
        cur.close()
        close_connection(conn)
    except Exception as e:
        print("Error:", str(e))