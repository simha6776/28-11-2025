# app.py
import os
import time
import threading
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2

from detector.yolo_detector import VehiclePlateDetector
from detector.tracker import TrackerWrapper
from detector.ocr import read_plate
from database import Session, events, shipments, stops, inventory,maintenance

from modules.route_optimizer import nearest_neighbor,two_opt,tour_length
from modules.predictive_maintenance import predict as predict_maint

from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy

import pandas as pd
from modules.predictive_maintenance import train_model

df = pd.read_csv('data/training_data.csv')
model = train_model(df)
print("✅ Model trained and saved successfully as models/maint_model.pkl")



app = Flask(__name__)
CORS(app)

MODEL_PATH = "yolov8n.pt"
MODEL_PATH_PLATE = "license_plate_detector.pt"
DETECT_CONF = 0.4

CAMERA_LOCATIONS = {
    "cam1": "Bengaluru Hub A",
    "cam2": "Mysuru Warehouse",
    "cam3": "Chennai Sorting Center",
    "cam4": "Mumbai Distribution Point",
    "cam5": "vtu mysuru",
}

#detector = YOLODetector(model_path=MODEL_PATH, conf=DETECT_CONF)
detector = VehiclePlateDetector()

tracker = TrackerWrapper()
processing = {}

def ensure_upload_folder():
    os.makedirs('uploads', exist_ok=True)

def process_video(filepath, camera_id="cam1"):
    sess = Session()
    cap = cv2.VideoCapture(filepath)
    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            # Detect vehicles
            detections = detector.detect(frame)
            sort_input = []
            for det in detections:
                 name = det.get("class_name", "")
                 text = det["plate_text"]
                 x1, y1, x2, y2 = det["coords"]
                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                 cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                 if name.lower() in ["car", "truck", "bus", "motorbike", "bike", "bicycle", "van"]:
                    x, y, w, h = det["bbox"]
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    score = det["conf"]
                    sort_input.append([x1, y1, x2, y2, score])

            trackers = tracker.update(sort_input)

            # Detect number plates separately
            plate_detections = detector.detect(frame)

            for t in trackers:
                x1, y1, x2, y2, tid = t
                w, h = x2 - x1, y2 - y1

                # Check if any license plate overlaps this vehicle
                plate_text = ""
                for p in plate_detections:
                    px1, py1, px2, py2 = p["bbox"]
                    # check overlap (basic IoU)
                    overlap_x1 = max(x1, px1)
                    overlap_y1 = max(y1, py1)
                    overlap_x2 = min(x2, px2)
                    overlap_y2 = min(y2, py2)
                    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                        crop = frame[int(py1):int(py2), int(px1):int(px2)]
                        plate_text = read_plate(crop)
                        break

                location_name = CAMERA_LOCATIONS.get(camera_id, "Unknown Location")
                sess.execute(events.insert().values(
                    object_id=str(int(tid)),
                    object_type="vehicle",
                    plate=plate_text,
                    camera_id=camera_id,
                    location=location_name,
                    timestamp=datetime.utcnow(),
                    frame=frame_no,
                    x=float(x1), y=float(y1), w=float(w), h=float(h)
                ))
            sess.commit()
    finally:
        cap.release()
        sess.close()
        processing.pop(filepath, None)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/supply_chain.html')
def supply_chain_page():
    return render_template('supply_chain.html')

@app.route('/inventory.html')
def inventory_page():
    return render_template('inventory.html')

@app.route('/maintenance.html')
def maintenance_page():
    return render_template('maintenance.html')

@app.route('/route_optimizer')
def route_optimizer_page():
    return render_template('route_optimizer.html')


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    cam = request.form.get('camera_id', 'cam1')
    if not f:
        return jsonify({"error":"no file"}), 400
    ensure_upload_folder()
    path = os.path.join('uploads', f.filename)
    f.save(path)
    t = threading.Thread(target=process_video, args=(path, cam), daemon=True)
    t.start()
    processing[path] = {"thread": t, "camera": cam, "started": time.time()}
    return jsonify({"status":"processing", "path": path})

@app.route('/events/search', methods=['GET'])
def search_events():
    q_plate = request.args.get('plate')
    sess = Session()
    query = events.select()
    if q_plate:
        query = query.where(events.c.plate.like(f"%{q_plate}%"))
    res = sess.execute(query).fetchall()
    out = []
    for r in res:
        out.append({
            "id": r.id,
            "object_id": r.object_id,
            "object_type": r.object_type,
            "plate": r.plate,
            "camera_id": r.camera_id,
            "location": r.location,
            "timestamp": str(r.timestamp),
            "frame": r.frame,
            "bbox": {"x": r.x, "y": r.y, "w": r.w, "h": r.h}
        })
    sess.close()
    return jsonify(out)

@app.route('/location', methods=['GET'])
def get_location():
    plate = request.args.get('plate')
    if not plate:
        return jsonify({"error": "Plate number required"}), 400
    sess = Session()
    query = events.select().where(events.c.plate.like(f"%{plate}%")).order_by(events.c.timestamp.desc()).limit(1)
    result = sess.execute(query).fetchone()
    sess.close()
    if result:
        return jsonify({
            "plate": result.plate,
            "last_seen": str(result.timestamp),
            "location": result.location,
            "camera_id": result.camera_id,
            "object_type": result.object_type
        })
    else:
        return jsonify({"message": "No record found for this plate"})
    
    # shipments CRUD
# 🚚 SHIPMENTS ENDPOINTS (working version)

from sqlalchemy import select

@app.route('/shipments', methods=['GET'])
def get_shipments():
    """List all shipments"""
    sess = Session()
    rows = sess.execute(select(shipments)).fetchall()
    sess.close()
    return jsonify([dict(r._mapping) for r in rows])


@app.route('/shipments', methods=['POST'])
def create_shipment():
    """Create a new shipment"""
    data = request.get_json() or {}
    sess = Session()
    ins = shipments.insert().values(
        shipment_code=data.get('code'),
        origin=data.get('origin'),
        destination=data.get('destination'),
        status='created',
        assigned_plate=data.get('assigned_plate')
    )
    sess.execute(ins)
    sess.commit()
    sess.close()
    return jsonify({'message': 'Shipment created successfully!'})


@app.route('/shipments/<int:sid>', methods=['GET'])
def get_shipment(sid):
    """Get single shipment by ID"""
    sess = Session()
    row = sess.execute(select(shipments).where(shipments.c.id == sid)).fetchone()
    sess.close()
    if not row:
        return jsonify({'error': 'Shipment not found'}), 404
    return jsonify(dict(row._mapping))

# predictive maintenance endpoint
@app.route('/predict_maintenance', methods=['POST'])
def predict_maintenance():
    data = request.get_json() or {}
    return jsonify(predict_maint(data))

# inventory endpoints
#@app.route('/inventory', methods=['GET', 'POST'])
#def inventory_list_create():
    #sess = Session()
    #if request.method == 'GET':
        #rows = sess.execute(inventory.select()).fetchall()
        #sess.close()
       # out = [dict(r) for r in rows]
        #return jsonify(out)
   # data = request.get_json() or {}
   #sess.execute(inventory.insert().values(sku=data.get('sku'), name=data.get('name'), qty=int(data.get('qty',0)), location=data.get('location',''))); sess.commit(); sess.close()
    #return jsonify({'status':'ok'})

#from modules import route_optimizer

#@app.route('/route_optimizer.html')
#    return render_template('route_optimizer.html')

#@app.route('/optimize_route', methods=['POST'])
#def optimize_route():
    #data = request.get_json()
    #points = data.get('points', [])
    #if not points or len(points) < 2:
        #return jsonify({'error': 'Need at least two points to optimize'}), 400

    #try:
       # tour = route_optimizer.nearest_neighbor(points)
        #improved = route_optimizer.two_opt(points, tour)
        #total_dist = route_optimizer.tour_length(points, improved)
       # return jsonify({'tour': improved, 'total_distance': round(total_dist, 2)})
    #except Exception as e:
        #return jsonify({'error': str(e)}), 500

from geopy.geocoders import Nominatim
from modules import route_optimizer
from flask import request, jsonify
import math

@app.route("/optimize_route", methods=["POST"])
def optimize_route():
    data = request.get_json()
    locations = data.get("locations")

    if not locations or len(locations) < 2:
        return jsonify({"error": "Need at least two locations"}), 400

    geolocator = Nominatim(user_agent="logistics-ai-tracking")
    points = []
    geo_info = []

    for loc in locations:
        geo = geolocator.geocode(loc)
        if geo:
            points.append([geo.latitude, geo.longitude])
            geo_info.append({
                "name": loc,
                "lat": geo.latitude,
                "lon": geo.longitude
            })
        else:
            return jsonify({"error": f"Could not locate {loc}"}), 400

    # Run optimization
    tour = route_optimizer.nearest_neighbor(points)
    optimized_points = [geo_info[i] for i in tour]
    total_distance = route_optimizer.tour_length(points, tour)

    return jsonify({
        "optimized_order": optimized_points,
        "total_distance_km": round(total_distance, 2)
    })

# app.py
#from flask import Flask, render_template, request, jsonify
#from modules.inventory_manager import InventoryManager
#from flask_cors import CORS

#inv = InventoryManager()

#@app.route("/")
#def home():
   # return render_template("inventory.html")

#@app.route("/inventory", methods=["GET"])
#def get_inventory():
    #return jsonify(inv.get_all_items())

#@app.route("/add", methods=["POST"])
#def add():
    #data = request.json
    #inv.add_item(data["name"], data["quantity"], data["price"], data["category"])
    #return jsonify({"message": "Item added successfully"})

#@app.route("/update/<int:item_id>", methods=["POST"])
#def update(item_id):
    #data = request.json
    #inv.update_item(item_id, data["name"], data["quantity"], data["price"], data["category"])
    #return jsonify({"message": "Item updated successfully"})

#@app.route("/delete/<int:item_id>", methods=["DELETE"])
#def delete_item(item_id):
    #inv.delete_item(item_id)
    #return jsonify({"message": "Item deleted successfully"})

#@app.route("/search", methods=["GET"])
#def search_item():
    #keyword = request.args.get("q", "")
   # return jsonify(inv.search_item(keyword))

#@app.route("/reset", methods=["POST"])
#def reset():
   # inv.reset_inventory()
    #return jsonify({"message": "Inventory reset successfully"})

#@app.route("/sale/<int:item_id>", methods=["POST"])
#def sale(item_id):
   # qty = request.json.get("qty", 0)
   # ok = inv.record_sale(item_id, qty)
    #if ok:
     #   return jsonify({"message": "Sale recorded"})
#    return jsonify({"error": "Not enough stock"}), 400

#@app.route("/alerts", methods=["GET"])
#def low_stock():
    #return jsonify(inv.low_stock_alerts())

from modules.inventory_manager import InventoryManager
inv = InventoryManager()

@app.route("/inventory_page")
def inventory_page_fixed():
    return render_template("inventory.html")

@app.route("/inventory", methods=["GET"])
def get_inventory():
    return jsonify(inv.get_all_items())

@app.route("/add", methods=["POST"])
def add_item():
    data = request.json
    inv.add_item(data["name"], data["quantity"], data["price"], data["category"])
    return jsonify({"message": "Item added successfully"})

@app.route("/update/<int:item_id>", methods=["POST"])
def update_item(item_id):
    data = request.json
    inv.update_item(item_id, data["name"], data["quantity"], data["price"], data["category"])
    return jsonify({"message": "Item updated successfully"})

@app.route("/delete/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    inv.delete_item(item_id)
    return jsonify({"message": "Item deleted successfully"})

@app.route("/search", methods=["GET"])
def search_inventory():
    keyword = request.args.get("q", "")
    return jsonify(inv.search_item(keyword))

@app.route("/reset", methods=["POST"])
def reset_inventory():
    inv.reset_inventory()
    return jsonify({"message": "Inventory reset successfully"})

@app.route("/sale/<int:item_id>", methods=["POST"])
def sale(item_id):
    qty = request.json.get("qty", 0)
    ok = inv.record_sale(item_id, qty)
    if ok:
        return jsonify({"message": "Sale recorded"})
    return jsonify({"error": "Not enough stock"}), 400

@app.route("/alerts", methods=["GET"])
def low_stock_alerts():
    return jsonify(inv.low_stock_alerts())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
