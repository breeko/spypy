import argparse
import datetime as dt
import time
import picamera
import json
from dateutil import parser
import shutil

import src.detect as detect

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--interval", type=int, default=1, help="interval in minutes to take photographs for object detection")
ap.add_argument("-b", "--begin", type=str, default=None, help="time to begin tracking (e.g. '10:00 AM')")
ap.add_argument("-e", "--end", type=str, default=None, help="time to end tracking (e.g. '9:00 PM')")
ap.add_argument("-l", "--logs", type=str, default="images/logs/", help="directory to store log of objects")
ap.add_argument("-d", "--images", type=str, default="images/", help="directory to store images")
ap.add_argument("-s", "--save", type=str, action="store_true", help="whether to store images in sub-directories")

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

def track(camera, interval, start_time, end_time, directory, save):
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
    now = dt.datetime.now()
    while now < end_time:
        next_picture_time = dt.datetime.now() + dt.timedelta(minutes=interval)
        camera.capture(TEMP_FILE_NAME)
        objects = detect.detect_from_image(TEMP_FILE_NAME)
        log_file_name = '{}/logs/{}.json'.format(directory, dt.datetime.now().strftime(TIME_FORMAT))
        
        with open(log_file_name, 'w') as fp:
            json.dump(objects, fp)
        
        if save:
            file_name = "{}.jpg".format(now.strftime(TIME_FORMAT))
            save_file_path = "{}/{}".format(directory, file_name)
            shutil.copy(TEMP_FILE_NAME, save_file_path)
                        
        print("Next picture will be taken at {}".format(next_picture_time.strftime(TIME_FORMAT)))
        
        seconds_to_sleep = (next_picture_time - dt.datetime.now()).total_seconds()
        seconds_to_sleep = max(0, seconds_to_sleep)
        time.sleep(seconds_to_sleep)
        now = dt.datetime.now()

if __name__ == "__main__":
    camera = setup_camera(vflip=args["vflip"], hflip=args["hflip"])

    if args["begin"] is None:
        begin_time = dt.datetime.now()
    else:
        begin_time = parser.parse(args["begin"])
    
    if args["end"] is None:
        end_time = dt.datetime.max
    else:
        end_time = parser.parse(args["end"])
        if end_time < begin_time:
            end_time + dt.timedelta(days=1)
    
    track(camera=camera, interval=args["interval"], start_time=begin_time, end_time=end_time, directory=args["dir"], save=args["save"])
