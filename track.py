import argparse
import datetime as dt
import time
import picamera
import json
from dateutil import parser

import detect

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--interval", type=int, default=1, help="interval in minutes to take photographs for object detection")
ap.add_argument("-s", "--start", type=str, default=None, help="time to begin tracking (e.g. '10:00 AM')")
ap.add_argument("-e", "--end", type=str, default=None, help="time to end tracking (e.g. '9:00 PM')")
ap.add_argument("-d", "--dir", type=str, default="logs/", help="directory to store log of objects")
ap.add_argument("--vflip", action="store_true", help="flip images taken from camera vertically")
ap.add_argument("--hflip", action="store_true", help="flip images taken from camera horizontally")

args = vars(ap.parse_args())

TEMP_FILE_NAME = "temp.jpg"
TIME_FORMAT = "%Y.%m.%d_%H.%M.%S"

def setup_camera(vflip, hflip):
    camera = picamera.PiCamera()
    camera.vflip = args["vflip"]
    camera.hflip = args["hflip"]
    return camera

def track(camera, interval, start_time, end_time, directory):
    detect.load_yolo_model()
    if start_time > dt.datetime.now():
        sleep_seconds = (start_time - dt.datetime.now()).seconds
        print("Starting at", start_time.strftime(TIME_FORMAT))
        time.sleep(sleep_seconds)

    if end_time is not None:
        print("Running until {}".format(end_time.strftime(TIME_FORMAT)))
    else:
        end_time = dt.datetime.max

    print("Starting...")
    while dt.datetime.now() < end_time:
        next_picture_time = dt.datetime.now() + dt.timedelta(minutes=interval)
        camera.capture(TEMP_FILE_NAME)
        objects = detect.detect_from_image(TEMP_FILE_NAME)
        file_name = '{}/{}.json'.format(directory, dt.datetime.now().strftime(TIME_FORMAT))

        with open(file_name, 'w') as fp:
            json.dump(objects, fp)

        print("Next picture will be taken at {}".format(next_picture_time.strftime(TIME_FORMAT)))
        
        seconds_to_sleep = (next_picture_time - dt.datetime.now()).total_seconds()
        seconds_to_sleep = max(0, seconds_to_sleep)
        time.sleep(seconds_to_sleep)

if __name__ == "__main__":
    camera = setup_camera(vflip=args["vflip"], hflip=args["hflip"])

    if args["start"] is None:
        start_time = dt.datetime.now()
    else:
        start_time = parser.parse(args["start"])
    
    if args["end"] is None:
        end_time = dt.datetime.max
    else:
        end_time = parser.parse(args["end"])
        if end_time < start_time:
            end_time + dt.timedelta(days=1)
    
    track(camera=camera, interval=args["interval"], start_time=start_time, end_time=end_time, directory=args["dir"])
