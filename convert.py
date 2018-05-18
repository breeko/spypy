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
ap.add_argument("-i", "--input", type=str, default=None, help="input image path on which to draw heatmap")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="alpha of each object detected. The higher the alpha the less transparent each object")
ap.add_argument("-b", "--object", type=str, default=None, help="object to draw on the heatmap")
ap.add_argument("-f", "--filter", type=str, default="", help="string filter to apply on input files. for exampled -f 2018.05.14 would only select input files that have 2018.05.14 in file name")

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
        obj_x, obj_y = obj["center"]
        _, _, obj_w, obj_h  = obj["xywh"]
        gauss = two_d_gaussian((x, y), obj_x, obj_y, obj_w, obj_h)
        cb = ax.contourf(x, y, gauss.reshape(x.shape[0], y.shape[1]), 14, cmap=mycmap, alpha=alpha)
    if len(objects) > 0:
        plt.colorbar(cb)
    
    return fig

def get_objects_from_directory(directory, object_type, filter_str, verbose=True):
    if directory[-1] == "/":
        directory = directory[:-1]
    
    objects = []

    num_frames_detected = 0
    num_objects_detected = 0
    num_files = 0

    failed = []

    for fp in listdir(directory):
        try:
            if fp[-5:].lower() == ".json" and filter_str in fp:
                num_files += 1
                json_dict = json.load(open("{}/{}".format(directory, fp)))
                objects_detected = json_dict.get(object_type, [])
                objects.extend(objects_detected)
                if len(objects_detected) > 0:
                    num_frames_detected += 1 
                    num_objects_detected += len(objects_detected)
        except json.JSONDecodeError:
            failed.append(fp)

    if verbose:
        print("{} files considered.\n{} frames contained {}.\n{} objects detected.".format(num_files, num_frames_detected, object_type, num_objects_detected))
        if len(failed) > 0:
            print("Number failed: {}".format(len(failed)))
            print(failed[:5])

    return objects

def draw_on_image_from_directory(directory, image, output, object_type, filter_str, alpha):
    if type(image) is str:
        image = Image.open(image)
    
    objects = get_objects_from_directory(directory, object_type, filter_str)
    if len(objects) > 0:
        fig = draw_on_image(image, objects, alpha)
        fig.savefig("{}".format(output))

        print("Saved file in {}".format(output))
    else:
        print("No {} found in logs".format(object_type))
        
if __name__ == "__main__":
    assert args["input"] is not None, "Invalid input"
    assert args["object"] is not None, "Invalid object to detect"
    assert args["alpha"] <= 1.0, "Alpha must be less than or equal to 1.0"
    assert args["alpha"] > 0.0, "Alpha must be greater than 0.0"

    draw_on_image_from_directory(args["dir"], args["input"], args["output"], args["object"], args["filter"], args["alpha"])    
