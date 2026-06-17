Proyect commit phases
1. initial commit with the retinaface basic module and resnet50
2. Implementation of RetinaFace with ONNX Runtime and MobileNetV3
3. Implementation of MiVOLO module and testing with the official body+face detector with native yolo8x
4. Finished rtdetr module for body detection and end of testing with basic pipeline
   * 4.1. The basic pipeline lets you add two different modules one for age estimator and other for face detection
5. Creation of advanced pipeline
    * 5.1 The advanced pipeline introduces a bifurcation of the data and enables the MiVOLO age estimator be fed by the body_crops made by rtdetr and the face crops of RetinaFace, for more accuret predictions
    * 5.2 Also now the RetinaFace makes the detections on the body_crop instead then the whole frame to be more optimal
