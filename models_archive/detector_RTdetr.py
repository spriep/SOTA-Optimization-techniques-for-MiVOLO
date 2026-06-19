import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

class RTDETRDetector:
    def __init__(self, weights_path="PekingU/rtdetr_r50vd", device=None):
        self.device = device 
        print(f"[*] Inicializando RT-DETR oficial desde {weights_path} en {self.device}...")
        
        # Loading the official processer and model
        self.processor = RTDetrImageProcessor.from_pretrained(weights_path)
        self.model = RTDetrForObjectDetection.from_pretrained(weights_path).to(self.device)
        self.model.eval()

    def detect(self, frame, conf_threshold=0.3):
        # Preprocesing: necessary for the model
        inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-procesing
        target_sizes = torch.tensor([(frame.shape[0], frame.shape[1])]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_threshold
        )
        
        parsed_detections = []
        for result in results:
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                # Filtrer only 'person' (ID 1 in COCO/RT-DETR)
                if self.model.config.id2label[label.item()] == "person":
                    box = box.tolist() # [x1, y1, x2, y2]
                    parsed_detections.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(score)])
        
        return parsed_detections