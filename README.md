# SpyPy
## Offline image detection and tracking intended for the Raspberry Pi
[Blog post explaining usage](https://medium.com/ml-everything/offline-object-detection-and-tracking-on-a-raspberry-pi-fddb3bde130)

[Blog post explaining YOLO model](https://medium.com/ml-everything/how-to-actually-easily-detect-objects-with-deep-learning-on-raspberry-pi-4fd40af84fee)

To track objects (take a picture every minute indefinitely, detect objects and save down object logs):

```
git clone https://github.com/breeko/spypy.git
cd spypy
./download_weights.sh # or just download weights manually
python track.py
```
To convert object logs to heatmap
```
python convert.py -i [input_image to draw over] -d [directory of logs] -o [object to detect]
```