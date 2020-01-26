import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cerradura(bin_img, iterations):
    copy = bin_img.copy()
    kernel = np.ones((3,3), np.uint8)
    dilatacion = cv2.dilate(copy, kernel, iterations=iterations)
    erosion = cv2.erode(dilatacion, kernel, iterations=iterations)
    return erosion

def openning(bin_img, iterations):
    copy = bin_img.copy()
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(copy, kernel, iterations=iterations)
    dilatacion = cv2.dilate(erosion, kernel, iterations=iterations)
    return dilatacion

def unsharp(image,K=0.6,blur_size=9):
    smooth=cv2.GaussianBlur(image,(blur_size,blur_size),10)
    g=cv2.addWeighted(image,1,smooth,-1,0)
    return cv2.addWeighted(image,1,g,K,0) #sharp


def maxCountour(contornos):
    # Encontrar el contorno de mayor area
    area=0
    for cnt in contornos:
        t_area = cv2.contourArea(cnt)
        if (t_area > area):
            area = t_area
            cntMax = cnt
    return (area,cntMax)

def getArea(frame):
    frame=unsharp(frame,0.9)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Obtener el paper
    ret,tresh=cv2.threshold(gray,75,255,cv2.THRESH_BINARY)
    opened=openning(tresh,6)   
    
    # Obtener los contornos detectados en la imagen abierta
    contornos, jerarquia = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    areaPapel=0
    cntMax=None
    (areaPapel,cntMax)=maxCountour(contornos)
    
    tmp=np.copy(frame)
    cv2.drawContours(tmp, [cntMax], 0, (255,0,0), 10)
    plt.subplot(121),plt.imshow(tmp)
    plt.title('Papel Delimitado')
    plt.xlabel('Anchura(px)')
    plt.ylabel('Altura(px)')
    plt.show()
    cv2.waitKey(0)
    
    leftmost = tuple(cntMax[cntMax[:,:,0].argmin()][0])
    rightmost = tuple(cntMax[cntMax[:,:,0].argmax()][0])
    topmost = tuple(cntMax[cntMax[:,:,1].argmin()][0])
    bottommost = tuple(cntMax[cntMax[:,:,1].argmax()][0])
    print ((leftmost,topmost), (rightmost,bottommost))
    
    not_openend = cv2.bitwise_not(opened[topmost[1]:bottommost[1],leftmost[0]:rightmost[0]], cv2.COLOR_BGR2HSV)
    close=cerradura(not_openend,20)

    plt.subplot(121),plt.imshow(close, cmap='gray')
    plt.title('Contorno de la Hoja')
    plt.xlabel('Anchura(px)')
    plt.ylabel('Altura(px)')
    plt.show()
    cv2.waitKey(0)

    # Obtener los contornos detectados en la imagen abierta
    contornos, jerarquia = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    areaHoja=0
    cntMax=None
    (areaHoja,cntMax)=maxCountour(contornos)
    
    realArea=(areaHoja*21.0*29.7*numPapel)/areaPapel
    print ("Area Leaf "+str(realArea))
    cv2.drawContours(frame[topmost[1]:bottommost[1],leftmost[0]:rightmost[0]], [cntMax], 0, (255,0,0), 3)

    plt.subplot(121),plt.imshow(frame[topmost[1]:bottommost[1],leftmost[0]:rightmost[0]])
    plt.title('Hoja Delimitada')
    plt.xlabel('Anchura(px)')
    plt.ylabel('Altura(px)')
    plt.show()
    cv2.waitKey(0)
    return realArea
"""
numPapel=16
folder='imageneshojasmaiz/segunda'
f= open("values.csv","w+")
f.write("name,fecha,area\n")
for x in os.listdir(folder):
    # Se muestra el resultado
    print(x)
    frame = cv2.imread(folder+'/'+x,-1)        #Captura una imagen de la camara y la guarda en la matriz frame
    area=getArea(frame)
    f.write(x+",cuarta,"+str(area)+"\n")

f.close() 
cv2.destroyAllWindows()

"""

"""numPapel=16
folder='imageneshojasmaiz/segunda'
frame = cv2.imread(folder+'/DSC_0053.JPG',-1)        #Captura una imagen de la camara y la guarda en la matriz frame
area=getArea(frame)
cv2.destroyAllWindows()
"""
# segunda\57 problema

#https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html
#https://docs.opencv.org/trunk/d4/d73/tutorial_py_contours_begin.html
#https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
# https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga2c759ed9f497d4a618048a2f56dc97f1 numero de pixeles