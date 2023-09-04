import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import cv2 as cv2
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib import cm 
from matplotlib.ticker import LinearLocator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy import ndimage, signal, misc

trackPoints = [[0,0]]
def onClick(event):
        if event.button is MouseButton.LEFT:
            print("click")
            print('button=%d, x=%d,y=%d, xdata=%f, ydata=%f' %
                (event.button, event.x, event.y, event.xdata, event.ydata))
            plt.plot(event.xdata,event.ydata, ',')
            trackPoints.append([event.xdata,event.ydata])
            plt.plot(event.xdata,event.ydata,'g^')
            fig.canvas.draw()
img = np.asarray(Image.open('monkmap.png'))
fig = plt.figure()
cid = fig.canvas.mpl_connect('motion_notify_event', onClick)
plt.imshow(img)
plt.show()
yes = input()
trackPoints.pop(0)
print(trackPoints)



#Lv = [[1,0],[1,1],[1,2],[1,3],[2,3],[3,3]]
MonkeyRange = 150
alpha = .5
beta = .5
h,w,c = img.shape
xRange = w
yRange = w
hStep = 5
xStep =int(xRange/hStep)
yStep = int(yRange/hStep)
#Lv =  [[i,np.sin(i)*i+5] for i in np.arange(0,xRange,hStep)]

Lv = trackPoints
Lvys = [0] * len(Lv)
Lvxs = [0] * len(Lv)

for i in range(len(Lv)):
    Lvxs[i] = Lv[i][0]
    Lvys[i] = Lv[i][1]

# Set up a figure twice as tall as it is wide
fig = plt.figure()
#fig.suptitle('A tale of 2 subplots')

# First subplot
ax = fig.add_subplot(2, 1, 1)

t1 = np.arange(0.0, len(Lv), 1)
t2 = np.arange(0.0, 3, 1)
ax.set_xlim(0, xRange)
ax.set_ylim(0,yRange)
ax.plot(Lvxs,Lvys)


monkPosVal = np.array([[0]*xStep for i in range(yStep)])
##monkPosVal = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]

point = 0
xIts = 0


for monkeyX in np.arange(0,xRange,hStep):
    yIts = 0
    for monkeyY in np.arange(0,yRange,hStep):
        #print("position:",monkeyX,monkeyY)
        LvInRange = [[0,0]] * len(Lv)
        it = 0
        dots = 0
        for k in range(len(Lv)):
            if np.sqrt(np.power((Lv[k][0] - monkeyX),2)+np.power(Lv[k][1]-monkeyY,2)) < MonkeyRange:
                LvInRange[it] = [Lv[k][0],Lv[k][1]]
                it=it+1
                #if k+1 < len(Lv):
                if k != 0:
                  
                    Aone = (Lv[k][0]-Lv[k-1][0])
                    Atwo =(Lv[k][1]-Lv[k-1][1])
                    
                    Bone = monkeyX-(Lv[k-1][0])
                    Btwo =monkeyY-Lv[k-1][1]
                    
                    #dots += np.abs(Aone * Bone + Atwo * Btwo)
                    
                    if int(np.abs(Aone * Bone + Atwo * Btwo)) > int(np.abs(np.linalg.norm([Aone,Atwo])*np.linalg.norm([Bone,Btwo]))-5) and int(np.abs(Aone * Bone + Atwo * Btwo)) < int(np.abs(np.linalg.norm([Aone,Atwo])*np.linalg.norm([Bone,Btwo]))+5):
                        dots += 1
                    
            
        ##monkPosVal[point] = it
        if xIts < monkPosVal.shape[0] and yIts < monkPosVal.shape[1] :
            monkPosVal[xIts][yIts] = alpha* dots + beta * it
        else:
            break
        yIts = yIts + 1
        point = point + 1
    xIts = xIts + 1



#ax = fig.add_subplot(2,1,2)
#fig, ax = plt.subplots()
X = np.arange(0,xRange,hStep)
Y = np.arange(0,yRange,hStep)
X,Y = np.meshgrid(X,Y)

#Z = monkPosVal[X][Y]
Z = np.array(monkPosVal)
#Z = np.rot90(Z,1)
print(Z)

#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
 #                      linewidth=0, antialiased=False)
#ax.set_zlim(-1,1)

cmap = plt.cm.coolwarm
my_cmap = cmap(np.arange(cmap.N))

# Set alpha
my_cmap[:,-1] = np.linspace(.5, .8, cmap.N)


# Create new colormap
my_cmap = ListedColormap(my_cmap)
pcm = ax.pcolormesh(Y,X,Z,cmap=my_cmap,shading='auto')
plt.gca().invert_yaxis()





plt.imshow(img)
#fig.colorbar(pcm,ax)
plt.show()
