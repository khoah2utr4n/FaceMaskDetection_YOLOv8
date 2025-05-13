from ultralytics import YOLO

def getModel(weightsPath='yolov8n.pt'):
    return YOLO(weightsPath) 


def trainModel(model, numEpochs, resume=False):
    results = model.train(data="data.yaml", epochs=numEpochs, imgsz=640, resume=resume)
    return results


def getPrediction(model, source):
    """Return a prediction of parsing model for parsing image
        Args:
            model (torch.nn.Module): Model used for prediction
            source (Path, PIL, np.ndarray): Path to the image or image to be predicted 
        Return:
            prediction (list): Prediction for parsing image
    """
    results = model.predict(source=source, conf=0.7, verbose=False)
    prediction = []
    for i in range(len(results[0].boxes.xywhn)):
        classIdx = results[0].boxes.cls[i].cpu().item()
        conf = round(results[0].boxes.conf[i].cpu().item(), 2)
        pred = list(results[0].boxes.xywhn[i].cpu().numpy())
        pred.append(classIdx)
        pred.append(conf)
        prediction.append(pred)
    return prediction