from flask import Flask, render_template, Response, jsonify, request, url_for
import cv2
from ultralytics import YOLO
import time
import os
import math
from waitress import serve
from PIL import Image
import json
import base64
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getstarted', methods=['GET', 'POST'])
def getstarted():
    return render_template('getstarted.html')

########################## WEB CAM DETECTION ############################
@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp/start', methods=['POST'])
def start_webcam():
    global webcam_status, counts_camera
    if webcam_status:
        return jsonify({'message': 'Webcam is already started'}), 400
    webcam_status = True
    counts_camera = [0, 0]
    return jsonify({'message': 'Webcam started successfully', 'newCounts': counts_camera}), 200

@app.route('/webapp/stop', methods=['POST'])
def stop_webcam():
    global webcam_status
    if not webcam_status:
        return jsonify({'message': 'Webcam is already stopped'}), 400
    webcam_status = False
    return jsonify({'message': 'Webcam stopped successfully'}), 200

@app.route('/webapp/get_counts_camera', methods=['GET'])
def get_counts_camera():
    global counts_camera
    return jsonify({'counts': counts_camera})

@app.route('/detect_palm_fruit', methods=['POST'])
def detect_palm_fruit():
    global counts_camera
    counts_camera = [0, 0]
    detection_results = [
        {"className": "Buah Matang", "count": counts_camera[0]},
        {"className": "Buah Mentah", "count": counts_camera[1]}
    ]
    return jsonify({"detectionResults": detection_results})

def generate_frames_web(path_x):
    global webcam_status, counts_camera
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    model = YOLO("model/kelapa.pt")
    classNames = ['buah matang', 'buah mentah']
    while webcam_status:
        success, img = cap.read()
        results = model(img, stream=True)

        counts_camera = [0, 0]
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if cls == 0:
                    counts_camera[0] += 1
                    color = (0, 255, 0)
                elif cls == 1:
                    color = (0, 0, 255)
                    counts_camera[1] += 1
                class_name = classNames[cls]
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=2, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()

####################### UPLOAD IMAGE ###########################
@app.route("/detect", methods=["POST"])
def detect():
    try:
        buf = request.files["image_file"]
        boxes = detect_objects_on_image(Image.open(buf.stream))
        return Response(
            json.dumps(boxes),
            mimetype='application/json'
        )
    except Exception as e:
        return Response(
            json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )
    
def detect_objects_on_image(buf):
    global counts_upload
    counts_upload = [0, 0]
    try:
        model = YOLO("model/best.pt")
        results = model(buf, conf=0.8)
        result = results[0]
        output = []
        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = box.cls[0].item()
            prob = round(box.conf[0].item(), 2)
            label = result.names[class_id]
            output.append([x1, y1, x2, y2, label, prob])
            if class_id == 0:
                counts_upload[0] += 1
            elif class_id == 1:
                counts_upload[1] += 1
        return output
    except Exception as e:
        print(f"Error during object detection: {e}")
        return [], 0, 0

@app.route('/webapp/get_counts_upload', methods=['GET'])
def get_counts_upload():
    global counts_upload
    return jsonify({'counts': counts_upload})

@app.route('/detect_upload', methods=['POST'])
def detect_upload():
    global counts_upload
    counts_upload = [0, 0]
    detection_results = [
        {"className": "Buah Matang", "count": counts_upload[0]},
        {"className": "Buah Mentah", "count": counts_upload[1]}
    ]
    return jsonify({"detectionResults": detection_results})

if __name__ == '__main__':
    webcam_status = False
    counts_camera = [0, 0]
    counts_upload = [0, 0]
    app.run(host='0.0.0.0', debug=True)
