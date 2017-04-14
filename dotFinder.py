from PIL import Image, ImageDraw
import random

# Gets the tuple locations of dots in the marked images
def findDots(filename):
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
    for i in range(im.width):
        for j in range(im.height):
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
    getNonExamples(filename, instances)
    #getCroppedExamples(filename, instances, classification)
    # saveMarkedImage(im, d, newName, classification, instances, colors)

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
    for i in range(len(instances)):
        counter = 1
        for j in instances[i]:
            cropped = im.crop( (j[0]-64, j[1]-64, j[0]+64, j[1]+64) )
            cropped.save("Examples/"+filename+classification[i]+str(counter)+".png")
            counter+=1

def getNonExamples(filename, instances):
    im = Image.open("Train/"+filename)
    width = im.width
    height = im.height
    subName = filename.replace(".jpg","")
    for z in range(200):
        x = random.random()*width
        y = random.random()*height
        isCopy = True
        for k in instances:
            for j in range(len(k)):
                if(abs(width-x-k[j][0]) < 100 and abs(height- y-k[j][1]) < 100):
                    isCopy = False
                    break
        if(not isCopy):
            z-=1
            continue
        cropped = im.crop( (x-64, y-64, x+64, y+64 ))
        cropped.save("Random/"+subName+"_"+str(z)+".png")
