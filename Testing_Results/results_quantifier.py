import numpy as np
import visualparams as viz
import cv2

costmap = np.load("/home/gabriel/PRE/bagfiles/images_extracted/costmaps1.npy")
costmapviz = np.zeros((viz.Y,viz.X,3), np.uint8)

min_cost = np.min(costmap)
max_cost = np.max(costmap)

for x in range(viz.X) :
    for y in range(viz.Y) :
        
        #If the cell is not empty because some cost has been generated
        if costmap[0,y,x]!= 0 :
            
            #Normalizing the content
            value = np.uint8(((costmap[0,y,x]-min_cost)/(max_cost-min_cost))*255)
            costmapviz[y,x] = (value, value, value)
        else :
            #If nothing we leave the image black
            costmapviz[y,x] = (0, 0, 0)
    #Applying the color gradient        
costmapviz = cv2.applyColorMap(src=costmapviz, colormap=cv2.COLORMAP_JET)
cv2.imshow("Costmap", cv2.resize(cv2.flip(costmapviz, 0),(viz.X*20,viz.Y*20)))
cv2.waitKey(0)
cv2.destroyAllWindows()