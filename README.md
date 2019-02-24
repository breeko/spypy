# SpyPy
## Offline image detection and tracking intended for the Raspberry Pi
[Blog post explaining usage](https://medium.com/ml-everything/offline-object-detection-and-tracking-on-a-raspberry-pi-fddb3bde130)

[Blog post explaining YOLO model](https://medium.com/ml-everything/how-to-actually-easily-detect-objects-with-deep-learning-on-raspberry-pi-4fd40af84fee)

To track objects (take a picture every minute indefinitely, detect objects and save down object logs):

```
git clone https://github.com/breeko/spypy.git
cd spypy
./download_weights.sh # or just download weights manually
python track.py -i [interval] -s [start] -e [end] -d [dir] --vflip --hflip

interval: is the interval in minutes in which you want a picture to be taken
start: is the time you want the camera to start (e.g. "10 am")
end: is the time you want the camera to end (e.g. "9 pm")
directory: is the directory that you want the json logs to be saved
vflip|hflip: is a flag that tells the camera to vertically|horizontally flip the pictures taken

```
To convert object logs to heatmap
```
python convert.py -i [input_image to draw over] -d [directory of logs] -o [object to detect]

dir: is the directory of the log (json) files that contain the object locations
output: is the output of the final image
input: is the base image to be drawn over
alpha: is the redness of the alpha of the circles detecting the objects (higher is darker)
object: is the object that you want to detect. If you have a lot of objects detected over a long period of time, you would want to set the alpha to be low, and vice versa.
```