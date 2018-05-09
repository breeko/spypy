import argparse
import datetime as dt
import time
import picamera
import json

from detect import detect_from_image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--interval", type=int, default=1, help="interval in minutes to take photographs for object detection")
ap.add_argument("-s", "--start", type=float, default=0, help="minutes before camera to begin")
ap.add_argument("-e", "--end", type=float, default=None, help="minutes to run the camera")
ap.add_argument("-d", "--dir", type=str, default="logs/", help="directory to store log of objects")
ap.add_argument("--vflip", action="store_true", help="flip images taken from camera vertically")
ap.add_argument("--hflip", action="store_true", help="flip images taken from camera horizontally")

args = vars(ap.parse_args())

TEMP_FILE_NAME = "temp.jpg"

def setup_camera(vflip, hflip):
    camera = picamera.PiCamera()
    camera.vflip = args["vflip"]
    camera.hflip = args["hflip"]
    return camera

def track(camera, interval, start, end, directory):
    if start > 0:
        print("Starting in", start, "minutes")
        time.sleep(start*60)

    if end is not None:
        end_time = dt.datetime.now() + dt.timedelta(minutes=end)
        print("Running until {}".format(end_time.strftime("%Y-%m-%d %H:%M:%S")))
    else:
        end_time = dt.datetime.max

    print("Starting...")
    while dt.datetime.now() < end_time:
        camera.capture(TEMP_FILE_NAME)
        objects = detect_from_image(TEMP_FILE_NAME)
        file_name = '{}/{}.json'.format(directory, dt.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))

        with open(file_name, 'w') as fp:
            json.dump(objects, fp)
    
        time.sleep(interval)

if __name__ == "__main__":
    camera = setup_camera(vflip=args["vflip"], hflip=args["hflip"])
    track(camera=camera, interval=args["interval"], start=args["start"], end=args["end"], directory=args["dir"])

