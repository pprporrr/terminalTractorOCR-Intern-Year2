from sklearn.cluster import KMeans
import numpy as np, pandas as pd, torch, time, cv2, os

def loadModel(modelPath):
    return torch.hub.load('/Users/ppr/Desktop/Project/e-GateOCR/yolov5_master', 'custom', path=modelPath, source='local')

def getImgFiles(directory):
    return [file for file in os.listdir(directory) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

def saveCroppedImg(img, boxes, outputDir, imgName, prefix=""):
    croppedImages = []
    for i, (_, box) in enumerate(boxes.iterrows()):
        xmin = int(box['xmin'])
        ymin = int(box['ymin'])
        xmax = int(box['xmax'])
        ymax = int(box['ymax'])
        
        croppedImg = img[ymin:ymax, xmin:xmax]
        
        paddedImg = np.ones((580, 580, 3), dtype=np.uint8) * 255
        h, w, _ = croppedImg.shape
        scale = min(580 / w, 580 / h)
        resizedImg = cv2.resize(croppedImg, None, fx=scale, fy=scale)
        dw = (580 - resizedImg.shape[1]) // 2
        dh = (580 - resizedImg.shape[0]) // 2
        paddedImg[dh:dh+resizedImg.shape[0], dw:dw+resizedImg.shape[1]] = resizedImg
        
        borderSize = 30
        paddedImg = cv2.copyMakeBorder(paddedImg, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        outputFilePath = os.path.join(outputDir, f"{imgName}_{prefix}{i + 1}.jpg")
        cv2.imwrite(outputFilePath, paddedImg)
        croppedImages.append(paddedImg)
    return croppedImages

def charboxesProcess(data, charBoxes):
    firstSet, secondSet = [], []
    for i in range(len(data)):
        if i < len(data) - 1:
            diff = data[i+1] - data[i]
            if diff > 100:
                X = np.array(data).reshape(-1, 1)
                kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(X)
                labels = kmeans.labels_
                firstSet = [data[i] for i in range(len(data)) if labels[i] == 0]
                secondSet = [data[i] for i in range(len(data)) if labels[i] == 1]
                break
            else:
                firstSet.append(data[i])
        else:
            firstSet.append(data[i])
    mergedSet = firstSet + secondSet
    sortedCharBoxes = charBoxes.sort_values(by='ymax', key=lambda x: x.map(dict(zip(mergedSet, range(len(mergedSet))))))
    
    return sortedCharBoxes

def processImg(inputDir, plateModel, charModel, plateOutputDir, charOutputDir):
    imgFiles = getImgFiles(inputDir)
    
    for imgFile in imgFiles:
        imgPath = os.path.join(inputDir, imgFile)
        imgName, extension = os.path.splitext(imgFile)
        
        img = cv2.imread(imgPath)
        if img is None:
            print(f"Unable to read image: {imgPath}")
            continue
        
        startTime = time.time()
        
        plateResults = plateModel(imgPath)
        boundingBoxes = plateResults.pandas().xyxy[0]
        plateBoxes = boundingBoxes[boundingBoxes['name'] == 'Plate']
        
        if not plateBoxes.empty:
            plateBoxes = plateBoxes.sort_values(by='confidence', ascending=False)
            plateBox = plateBoxes.iloc[0]
            plateImg = saveCroppedImg(img, pd.DataFrame([plateBox]), plateOutputDir, imgName, prefix="Plate")
            
            for i, plateCroppedImg in enumerate(plateImg):
                plateImgName = f"{imgName}"
                charResults = charModel(plateCroppedImg)
                charboundingBoxes = charResults.pandas().xyxy[0]
                charBoxes = charboundingBoxes[charboundingBoxes['name'] == 'Character']
                if not charBoxes.empty:
                    charBoxes = charBoxes.sort_values(by='xmin', ascending=True)
                    charData = charBoxes['ymax'].tolist()
                    sortedCharBoxes = charboxesProcess(charData, charBoxes)
                    saveCroppedImg(plateCroppedImg, sortedCharBoxes, charOutputDir, plateImgName, prefix="Char")
        else:
            print(f"No plate detected in image: {imgPath}")
        
        endTime = time.time()
        processingTime = endTime - startTime
        
        print(f"Processing time for image '{imgFile}': {processingTime:.4f} seconds")

if __name__ == "__main__":
    inputDir = "/Users/ppr/Desktop/Project/e-GateOCR/ImgTest"
    plateOutputDir = "/Users/ppr/Desktop/Project/e-GateOCR/plateImgTest"
    charOutputDir = "/Users/ppr/Desktop/Project/e-GateOCR/characterImgTest"
    plateModelPath = '/Users/ppr/Desktop/Project/e-GateOCR/Models/bestPlateDetection.pt'
    charModelPath = '/Users/ppr/Desktop/Project/e-GateOCR/Models/bestCharDetection.pt'
    
    plateModel = loadModel(plateModelPath)
    charModel = loadModel(charModelPath)
    
    if not os.path.exists(plateOutputDir):
        os.makedirs(plateOutputDir)
    if not os.path.exists(charOutputDir):
        os.makedirs(charOutputDir)
    
    processImg(inputDir, plateModel, charModel, plateOutputDir, charOutputDir)