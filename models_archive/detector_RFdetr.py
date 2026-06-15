import torch
from rfdetr import RFDETRMedium, RFDETRLarge
import supervision as sv

class RFDETRDetector:
    def __init__(self, weights_path="medium", device=None):
        """
        Wrapper for the official Roboflow RF-DETR model. selecting size
        """
        self.device = device         
        # We load the corresponding class based on the Roboflow chart --> we could add more sizes
        if weights_path and "large" in weights_path.lower():
            print(f"[*] Initializing Official RF-DETR Large on {self.device}...")
            self.model = RFDETRLarge()
        else:
            print(f"[*] Initializing Official RF-DETR Medium on {self.device}...")
            self.model = RFDETRMedium()
    
    def detect(self, frame, conf_threshold=0.5):
        """
        Receive an OpenCV frame and return the PERSON boxes.
        """
        # Run official prediction (returns a supervision.Detections object)
        detections = self.model.predict(frame, threshold=conf_threshold)
        
        # The boxes you find are labeled person/car... we only want the boxes for people id=1 (COCO dataset)
        person_mask = detections.class_id == 1
        person_detections = detections[person_mask]
        
        # Extracting coordinates with pur pipelines format[x1, y1, x2, y2, score]
        parsed_detections = []
        
        # supervision.Detections automatically saves boxes in xyxy format
        for xyxy, confidence in zip(person_detections.xyxy, person_detections.confidence):
            x1, y1, x2, y2 = xyxy
            parsed_detections.append([int(x1), int(y1), int(x2), int(y2), float(confidence)])
            
        return parsed_detections

