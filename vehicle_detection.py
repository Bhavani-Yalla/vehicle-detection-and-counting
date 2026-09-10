import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11s.pt")

# Open video
cap = cv2.VideoCapture("road.mp4")

if not cap.isOpened():
    print("Error: Could not open road.mp4")
    exit()

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Vehicles already counted
counted_ids = set()

# Counts
vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (960, 540))

    height, width = frame.shape[:2]

    # -----------------------------
    # COUNTING LINE
    # -----------------------------
    line_y = int(height * 0.60)

    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (255, 0, 0),
        3
    )

    # -----------------------------
    # TRACK VEHICLES
    # -----------------------------
    results = model.track(
        frame,
        persist=True,
        conf=0.30,
        imgsz=640,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml",
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes

        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):

            class_id = int(box.cls[0])

            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Center of vehicle
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            vehicle_name = vehicle_classes[class_id]

            # -----------------------------
            # COUNT WHEN CROSSING LINE
            # -----------------------------
            if center_y > line_y and track_id not in counted_ids:

                counted_ids.add(track_id)

                vehicle_counts[vehicle_name] += 1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw center
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Label
            label = f"{vehicle_name} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # -----------------------------
    # DISPLAY COUNTS
    # -----------------------------

    total = sum(vehicle_counts.values())

    cv2.putText(
        frame,
        f"Cars: {vehicle_counts['Car']}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {vehicle_counts['Motorcycle']}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {vehicle_counts['Bus']}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {vehicle_counts['Truck']}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"TOTAL: {total}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # Show video
    cv2.imshow(
        "Vehicle Detection and Counting",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL COUNT")
print("-----------")
print("Cars:", vehicle_counts["Car"])
print("Motorcycles:", vehicle_counts["Motorcycle"])
print("Buses:", vehicle_counts["Bus"])
print("Trucks:", vehicle_counts["Truck"])
print("TOTAL:", sum(vehicle_counts.values()))