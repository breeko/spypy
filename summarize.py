import argparse
import json
from os import listdir
from pandas import DataFrame

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dir", type=str, default="logs/", help="directory to source json files with objects")
ap.add_argument("-f", "--filter", type=str, default="", help="string filter to apply on input files. for exampled -f 2018.05.14 would only select input files that have 2018.05.14 in file name")

args = vars(ap.parse_args())

def summarize_logs(directory, filter_str):
    if directory[-1] == "/":
        directory = directory[:-1]
    
    objects = {}

    for fp in listdir(directory):
        if fp[-5:].lower() == ".json" and filter_str in fp:
            json_dict = json.load(open("{}/{}".format(directory, fp)))
            for k, v in json_dict.items():
                if type(v) is not list:
                    # meta field like time
                    continue
                    
                if k not in objects:
                    objects[k] = {}
                num_files = objects[k].get("num_files", 0) + 1
                num_objects = objects[k].get("num_objects", 0) + len(v)
                objects[k] = {"num_files": num_files, "num_objects": num_objects}
    if len(objects) == 0:
        if filter_str == "":
            print("No files detected in {}".format(directory))
        else:
            print("No files detected in {} using filter {}".format(directory, filter_str))
    else:
        df = DataFrame(objects).transpose()
        print(df.sort_values("num_files", ascending=False))
        

if __name__ == "__main__":
    summarize_logs(args["dir"], args["filter"])    
