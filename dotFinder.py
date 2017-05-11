from PIL import Image, ImageDraw
import random
import numpy
import os

# The number of pixels we want our window to look at centered on the sealion
EXAMPLE_SIZE = 64 # change to 128? to 40?
# The fraction of each example size to overlap the window by
OVERLAP = 3/4

# Gets the tuple locations of dots in the marked images
def findDots(filename):
    print(filename)
    if(not "/" in filename):
        filename = "TrainDotted/"+filename
    newName = filename.replace(".jpg","marked.png")
    im = Image.open(filename)
    d = ImageDraw.Draw(im)
    classification = ["Male", "Female", "Juvenile", "Pup", "Subadult Male"]
    colors = [(100,250,150), (200,100,150), (200,200,80), (250,200,200), (80, 80, 255)]
    instances = []
    for i in range(5):
        instances.append([])
    for i in range(0, im.width, 5):
        for j in range(0, im.height, 5):
            color = im.getpixel((i,j))
            # If the pixel is red, then could be on adult male
            if(color[0] > 240 and color[1] < 30 and color[2] < 30):
                if(not inCurrentData(i,j,0, instances)):
                    instances[0].append((i,j))
            # If the pixel is pink, then could be on female
            elif(color[0] > 80 and color[0] < 100 and color[1] > 35 and color[1] < 53 and color[2] < 18):
                if(not inCurrentData(i,j,1, instances)):
                    instances[1].append((i,j))
            # If the pixel is brown, then could be on juvenile
            elif(color[0] < 35 and color[1] > 55 and color[1] < 70 and color[2] > 165):
                if(not inCurrentData(i,j,2, instances)):
                    instances[2].append((i,j))
            # If the pixel is green, then could be on pup
            elif(color[0] < 45 and color[1] > 160 and color[2] < 30):
                if(not inCurrentData(i,j,3, instances)):
                    instances[3].append((i,j))
            # If the pixel is blue, then could be on subadult male
            elif(color[0] > 230 and color[1] < 20 and color[2] > 230):
                if(not inCurrentData(i,j,4, instances)):
                    instances[4].append((i,j))
    filename = filename.replace("TrainDotted/","")
    getNonExamples(filename, instances, 400)
    getCroppedExamples(filename, instances, classification)
    # saveMarkedImage(im, d, filename, classification, instances, colors)

# Checks to see if the x,y pair is within some range of another occurance
# in the same instance. Serves the purpose of making sure a dot is only counted
# once
def inCurrentData(x, y, colorIndex, instances):
    currentList = instances[colorIndex]
    xRange = 10
    yRange = 10
    for i in range(len(currentList)):
        if(abs(x-currentList[i][0]) < xRange and abs(y-currentList[i][1]) < yRange):
            return True
    return False

# Saves the image with boxes around found boxes
def saveMarkedImage(im, d, filename, classification, instances, colors):
    newName = "123"+filename
    for j in range(len(instances)):
        print(classification[j]+": "+str(len(instances[j])))
        for i in instances[j]:
            d.rectangle([i[0]-5, i[1]-5, i[0]+5, i[1]+5], outline=colors[j])
    im.save(newName)

# Crops all the found instances of markings in the Dotted images into usable
# examples for our classifier. Saves into a seperate example folder
def getCroppedExamples(filename, instances, classification):
    filename = filename.replace("TrainDotted/","")
    im = Image.open("Train/"+filename)
    filename = filename.replace(".jpg","")
    counter = 1
    padding = EXAMPLE_SIZE/2
    for i in range(len(instances)):
        for j in instances[i]:
            cropped = im.crop( (j[0]-padding, j[1]-padding, j[0]+padding, j[1]+padding) )
            cropped.save("Examples/sealion/"+classification[i]+"_"+filename+str(counter)+".jpg")
            counter+=1

# filename is the file to extract from
# instances is an array of the locations of every sea lion in the image
# numberNons is the number of example to extract
def getNonExamples(filename, instances, numberNons):
    im = Image.open("Train/"+filename)
    width = im.width
    height = im.height
    subName = filename.replace(".jpg","")
    for z in range(numberNons):
        x = random.random()*width
        y = random.random()*height
        isCopy = False
        # we check if our random x,y pair is within 100 pixels of a sea lion
        for k in instances:
            for j in range(len(k)):
                if(abs(x-k[j][0]) < 100 and abs(y-k[j][1]) < 100):
                    isCopy = True
                    break
        if(isCopy):
            z-=1
        else:
            cropped = im.crop( (x-64, y-64, x+64, y+64 ))
            cropped.save("Examples/background/"+subName+"_"+str(z)+".png")


def getImageArray(filename):
    # Might need to edit the spacing, could be something we change manually
    # maybe another time where we need to teach the right spacing /s
    im = Image.open(filename)
    # fraction is the fraction of the total sample size to move by.
    padding = int(EXAMPLE_SIZE/2)
    spacing = int(EXAMPLE_SIZE*OVERLAP)
    width = im.width
    height = im.height
    examplesArray = []

    print(padding)
    print(str((width-2*padding)/spacing))
    print(str((height-2*padding)/spacing))
    print(spacing)

    for i in range(padding, width-padding, spacing):
        for j in range(padding, height-padding, spacing):
            c = im.crop((i-padding, j-padding, i+padding, j+padding))
            examplesArray.append(c)
    return examplesArray

# Takes in an array of image data and returns a list of tensors to use as
# input for a neural network
def imageArrayToTensor(array):
    b = []
    counter = 1
    for i in array:
        if(counter%int(len(array)/10) == 0):
            print(".", end = "")
        img = i.convert('RGB')
        x = numpy.asarray(img, dtype='float32')
        # x = numpy.expand_dims(x, axis=0)
        b.append(x)
        counter+=1
    return b

# Draws the resulting positive and negative locations from the neural network
# onto the original image
def drawOutputGrid(results, im, id):
    d = ImageDraw.Draw(im)
    padding = int(EXAMPLE_SIZE/2)
    spacing = int(EXAMPLE_SIZE*OVERLAP)
    width = int(im.width - padding)
    height = int(im.height - padding)
    counter = 0
    columnShift = int((height-padding)/spacing)
    for i in range(padding, width, spacing):
        for j in range(padding, height, spacing):
            countingValue = int(((i-padding)/spacing)*columnShift+((j-padding)/spacing))
            counter+=1
            if(results[countingValue] > 0.95):
                d.rectangle(((i-padding), (j-padding), (i+padding), (j+padding)), fill = (255,255,255))
    im.save("Output"+id+".jpg")

def getExamples():
    filenames = os.listdir("TrainDotted")
    for name in filenames:
        findDots(name)
