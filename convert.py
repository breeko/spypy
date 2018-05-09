import argparse
import json
from os import listdir
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", type=str, default="logs/", help="directory to source json files with objects")
ap.add_argument("-o", "--output", type=str, default="detected.png", help="output image path")
ap.add_argument("-i", "--input", type=float, default=None, help="input image path on which to draw heatmap")
ap.add_argument("-a", "--alpha", type=int, default=0.01, help="alpha of each object detected. The higher the alpha the less transparent each object")
ap.add_argument("-b", "--object", type=str, default=None, help="object to draw on the heatmap")

args = vars(ap.parse_args())

#2D Gaussian function
def two_d_gaussian(xy, xo, yo, sigma_x, sigma_y):
    x, y = xy
    a = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    c = 1./(2*sigma_x**2) + 1./(2*sigma_y**2)
    g = np.exp( - (a*((x-xo)**2) + c*((y-yo)**2)))
    return g.ravel()


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

def draw_on_image(image, objects, alpha):
    #Use base cmap to create transparent
    mycmap = transparent_cmap(plt.cm.Reds)

    # Import image and get x and y extents
    p = np.asarray(image).astype('float')

    w, h = image.size
    y, x = np.mgrid[0:h, 0:w]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)

    for obj in objects:
        obj_x,obj_y = obj["center"]
        _,_,obj_w,obj_h  = obj["xywh"]
        gauss = two_d_gaussian((x, y), obj_x, obj_y, obj_w, obj_h)
        cb = ax.contourf(x, y, gauss.reshape(x.shape[0], y.shape[1]), 14, cmap=mycmap, alpha=0.1)
    
    if len(objects) > 0:
        plt.colorbar(cb)
    
    return fig

if __name__ == "__main__":
    assert args["input"] is not None, "Invalid input"
    assert args["object"] is not None, "Invalid object to detect"
    assert args["alpha"] <= 1.0, "Alpha must be less than or equal to 1.0"
    assert args["alpha"] > 0.0, "Alpha must be greater than 0.0"

    image = Image.open(args["input"])
    
    objects = []

    directory = args["dir"]
    
    if directory[-1] == "/":
        directory = directory[:-1]
    
    num_frames_detected = 0
    num_objects_detected = 0
    
    for fp in listdir(args["dir"]):
        if fp[-5:].lower() == ".json":
            json_file = json.load(open("{}/{}".format(directory, fp)))
            objects_detected = json_file.get(args["object"], [])
            objects.extend(objects_detected)
            if len(objects) > 0:
                num_frames_detected += 1 
                num_objects_detected += len(objects)        
    
    fig = draw_on_image(image, objects, args["alpha"])
    fig.savefig("{}".format(args["output"]))
    print("{} frames contained {}. {} objects detected.".format(num_frames_detected, args["object"], num_objects_detected))
    print("Saved file in {}".format(args["output"]))
    

