from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model("person.jpg")

for box in results[0].boxes:
    cls = model.names[int(box.cls)]
    print(cls)
    print(type(cls))