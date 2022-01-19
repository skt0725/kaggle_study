import cv2, dlib, sys
import numpy as np
from imutils import face_utils, resize

scalar = 0.3

cap = cv2.VideoCapture('face.mp4')
orange_img = cv2.imread('orange.png')
orange_img = cv2.resize(orange_img, dsize=(400, 400))
detector =dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while cap.isOpened():
    ret, img = cap.read()
    
    if not ret:
        break
        
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    faces = detector(img)
    result = orange_img.copy()
    if len(faces) == 0:
        continue
    face = faces[0]

    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    face_img = img[y1:y2, x1:x2].copy()

    shape = predictor(img, face)
    shape = face_utils.shape_to_np(shape)
    
    for p in shape:
        cv2.circle(face_img, center = (p[0] - x1, p[1] - y1), radius=2, color = 255, thickness=1)

    # eyes
    le_x1 = shape[36,0]
    le_y1 = shape[37,1]
    le_x2 = shape[39,0]
    le_y2 = shape[41,1]
    le_margin = int((le_x2 - le_x1) * 0.18)
    
    re_x1 = shape[42, 0]
    re_y1 = shape[43, 1]
    re_x2 = shape[45, 0]
    re_y2 = shape[47, 1]
    re_margin = int((re_x2 - re_x1) * 0.18)
    
    left_eye_img = img[le_y1-le_margin:le_y2+le_margin, le_x1-le_margin:le_x2+le_margin].copy()
    right_eye_img = img[re_y1-re_margin:re_y2+re_margin, re_x1-re_margin:re_x2+re_margin].copy()
    
    left_eye_img = resize(left_eye_img, width=100)
    right_eye_img = resize(right_eye_img, width=100)
    
    result = cv2.seamlessClone(
        left_eye_img,
        result,
        np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
        (120, 150),
        cv2.MIXED_CLONE
    )
    result = cv2.seamlessClone(
        right_eye_img,
        result,
        np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
        (280, 150),
        cv2.MIXED_CLONE
    )
    mouth_x1 = shape[48, 0]
    mouth_y1 = shape[50, 1]
    mouth_x2 = shape[54, 0]
    mouth_y2 = shape[57, 1]
    mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)
    
    mouth_img = img[mouth_y1-mouth_margin:mouth_y2+mouth_margin, mouth_x1-mouth_margin:mouth_x2+mouth_margin].copy()
    mouth_img = resize(mouth_img, width=250)
    
    result = cv2.seamlessClone(
        mouth_img,
        result,
        np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
        (200, 280),
        cv2.MIXED_CLONE
    )
    cv2.imshow('face', face_img)
    cv2.imshow('result', result)
    cv2.waitKey(1)
cv2.destroyAllWindows()
    
    
    
    