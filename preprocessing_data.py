import os
import xml.etree.ElementTree as ET 
import config
from utils import move_file, xml2YoloBox
from sklearn.model_selection import train_test_split 


def createDatasetConfigFile():
    root = os.getcwd()
    datasetPath = os.path.join(root, config.DATASET_DIR)
    
    # Create a config file for dataset (dataset.yaml)
    yaml_text = f"""path: {datasetPath}
train: images/train 
val: images/val/ 
test: images/test/

names:
    0: without_mask
    1: with_mask
    2: mask_weared_incorrect"""

    with open("data.yaml", 'w') as file:
        file.write(yaml_text)
    

def xml2YoloFormat(filepath):
    """Convert all objects in xml file to Ultralytics YOLO format.
    
        Args:
            filepath (Path): Path to xml file 
        Returns:
            allObjs (list): All objects in Ultralytics YOLO format
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    imageWidth = int(root.find('size').find('width').text)    
    imageHeight = int(root.find('size').find('height').text)
    
    allObjs = []
    for obj in root.findall('object'):
        className = obj.find('name').text
        classIdx = config.CLASS_INDEXS[className]
        xmlBox = [int(obj.find('bndbox')[i].text) for i in range(4)]
        yoloBox = xml2YoloBox(xmlBox, imageWidth, imageHeight)
        allObjs.append([classIdx] + yoloBox)
    return allObjs


def createLabels(annotationsDir, labelsDir):
    """
        Creates label files (.txt) in YOLO format for each annotation files (.xml)
    """
    os.makedirs(labelsDir, exist_ok=True)
    for filename in config.ALL_FILENAMES:
        xml_filepath = os.path.join(annotationsDir, filename) + '.xml'
        txt_filepath = os.path.join(labelsDir, filename) + '.txt'
        data = xml2YoloFormat(xml_filepath)
        with open(txt_filepath, 'w') as f:           
            f.write('\n'.join(' '.join(map(str, obj)) for obj in data))
            f.close() 
    

if __name__ == '__main__':
    # Create label directory and convert annotations
    print("Creating labels ...")
    createLabels(config.ANNOTATIONS_DIR, config.LABELS_DIR)
    
    #  Split data to train/val/test sets
    random_state = 1
    train, valTest = train_test_split(config.ALL_FILENAMES, test_size=0.3, 
                                       random_state=random_state, shuffle=True) 
    val, test = train_test_split(list(valTest), test_size=0.5, 
                                 random_state=random_state, shuffle=True)

    # Move image and label files to corresponding train/val/test directories
    move_file(train, config.IMAGES_DIR, f'{config.IMAGES_DIR}/train/', 
              config.LABELS_DIR, f'{config.LABELS_DIR}/train/')
    move_file(val, config.IMAGES_DIR, f'{config.IMAGES_DIR}/val/', 
              config.LABELS_DIR, f'{config.LABELS_DIR}/val/')
    move_file(test, config.IMAGES_DIR, f'{config.IMAGES_DIR}/test/', 
              config.LABELS_DIR, f'{config.LABELS_DIR}/test/')
    
    # Create a dataset config file
    print('Creating dataset config file ...')
    createDatasetConfigFile()
    
    print('Done!!')
    