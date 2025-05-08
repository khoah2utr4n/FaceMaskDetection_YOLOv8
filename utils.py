import matplotlib.pyplot as plt
import shutil
import cv2
import os
import config


def xml2YoloBox(bndbox, width, height):
    """Convert xml bounding box to YOLO bounding box.
    
        Args:
            bndbox (list | np.darray): A xml bounding box with format [xmin, ymin, xmax, ymax]
            width (int): A width of entire image
            height (int): A height of entire image
        Returns:
            yoloBox (list): The bounding box in YOLO format [xcenter, ycenter, boxWidth, boxHeight]
    """
    xcenter = ((bndbox[0] + bndbox[2]) / 2.) / width
    ycenter = ((bndbox[1] + bndbox[3]) / 2.) / height
    boxWidth = (bndbox[2] - bndbox[0]) / width
    boxHeight = (bndbox[3] - bndbox[1]) / height
    yoloBox = [xcenter, ycenter, boxWidth, boxHeight]
    return yoloBox


def yolo2XmlBox(bndbox, width, height):
    """Convert YOLO bounding box to xml bounding box.
    
        Args:
            bndbox (list | np.darray): A YOLO bounding box with format [xcenter, ycenter, boxWidth, boxHeight]
            width (int): A width of entire image
            height (int): A height of entire image
        Returns:
            xmlBox (list): The bounding box in xml format [xmin, ymin, xmax, ymax]
    """
    xmin = (bndbox[0] - bndbox[2] / 2.) * width
    ymin = (bndbox[1] - bndbox[3] / 2.) * height
    xmax = (bndbox[0] + bndbox[2] / 2.) * width
    ymax = (bndbox[1] + bndbox[3] / 2.) * height
    xmlBox = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return xmlBox


def moveFile(filenames, imgPath, imgDest, labelPath, labelDest):
    os.makedirs(imgDest, exist_ok=True)    
    os.makedirs(labelDest, exist_ok=True)

    for filename in filenames:
        imgSrc = os.path.join(imgPath, filename + '.png')
        labelSrc = os.path.join(labelPath, filename + '.txt')
        shutil.move(imgSrc, imgDest)
        shutil.move(labelSrc, labelDest)


def drawBoxes(image, bndboxes, withConfScore=False, isRgb=True):
    """Draw parsing bounding boxes on an parsing image.
        Args:
            image (Image): The original image.
            bndboxes (list): List of predicted bounding boxes, format: [x, y, w, h, cls, conf].
            name (str): Name to save the image.
            withConfScore (bool, optional): Show confidence score or not. Defaults is False.
            isRgb (bool, optional): The parsing image is rgb or bgr? (Just to keep the bounding box color consistent).
        Returns:
            (Image): The image with drawn bounding boxes.
    """
    # Specific color for each class
    if isRgb:
        classColor = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255)}
    else: # bgr
        classColor = {2: (255,0,0), 1: (0,255,0), 0: (0,0,255)}
        
    
    # Load the image
    newImg = image.copy()
    h, w, _ = newImg.shape
  
    for obj in bndboxes:
        xmin, ymin, xmax, ymax = yolo2XmlBox(obj[:4], w, h)
        classIdx = obj[4]
        className = config.CLASS_NAMES[classIdx]
        color = classColor[classIdx]
        text = f"{className}({obj[5]})" if withConfScore else f"{className}"
        
        newImg = cv2.rectangle(newImg, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
        newImg = cv2.putText(newImg, text, (xmin, ymin-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
    return newImg


def showImage(imagePath, predictedBoxes=None, labelPath=None):
    """Display an image with optinal predicted bounding boxes and true bounding boxes
    
        Args:
            imagePath (Path): Path to image
            predictedBoxes (list | np.darray, optinal): 
            labelPath (str, optinal): Path to true bounding boxes. Default is None
    """
    # Create a figure for plotting
    fig = plt.figure(figsize=(12, 8))
    numRows = 1
    numCols = 3 if (predictedBoxes is not None and labelPath is not None) else 2  
    
    # Load the image
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    # Display the original image
    imgIdx = 1
    ax1 = plt.subplot(numRows, numCols, imgIdx)
    ax1.imshow(image)
    ax1.set_title('Original image')
    
    # Display the predicted bounding boxes
    if predictedBoxes is not None:
        imgIdx += 1
        ax2 = plt.subplot(numRows, numCols, imgIdx)
        predictedImg = drawBoxes(image, predictedBoxes, withConfScore=True)
        ax2.imshow(predictedImg)
        ax2.set_title('Prediction')
    
    
    # Display the true bouding boxes
    if labelPath is not None:
        imgIdx += 1
        ax3 = plt.subplot(numRows, numCols, imgIdx)
        
        # Load true bounding boxes from label file
        trueBoxes = []
        with open(labelPath) as labelFile:
            for line in labelFile.readlines():
                bndbox = list(map(float, line.split()))
                order = [1, 2, 3, 4, 0]
                bndbox = [bndbox[order[i]] for i in range(5)]
                trueBoxes.append(bndbox)
        
        groundtruthImg = drawBoxes(image, trueBoxes, withConfScore=False)
        ax3.imshow(groundtruthImg)
        ax3.set_title('Grouth truth')
    fig.tight_layout()
    plt.show()