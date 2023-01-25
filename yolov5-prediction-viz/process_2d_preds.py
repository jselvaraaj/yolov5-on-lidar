import ast
from collections import defaultdict
import csv
import math
import pickle
import viz_utils as utils

def process_scan_factory(preds):
    def process_scan(scan):
        nonlocal preds
        bboxs = []
        
        for pred in preds[scan.frame_id]:
            x,y,w,h = pred['xywh']
            
            if math.isclose(w,0) or math.isclose(h,0):
                continue

            cuboids,bbox2D = scan.process_pred(x,y,w,h)

            bboxs.extend(cuboids)
            bboxs.append(bbox2D)

        return bboxs

    return process_scan

def read_preds(csv_path):
    preds = defaultdict(list)
    with open(csv_path,"r") as f:
        for line in csv.DictReader(f):
            line['xywh'] = ast.literal_eval(line['xywh'])
            preds[int(line['frame'])].append(line)
    return preds
    

def start_process(pcap_path, meta_path, csv_path):
    pcap_path = str(pcap_path)
    meta_path = str(meta_path)
    csv_path = str(csv_path)

    preds = read_preds(csv_path)
            
    process_scan = process_scan_factory(preds)
    bboxs_in_pcap = dict()

    pcap = utils.PcapWrapper(pcap_path,meta_path)
    print("started processing")
    for i,scan in enumerate(pcap):
        if len(preds[scan.frame_id]) != 0:
            print("\nPredctions in frame ", scan.frame_id)
            bboxs = process_scan(scan, preds)
            bboxs_in_pcap[scan.frame_id] = bboxs
            print("\tMoving on to the next frame")

    print("Finished processing")

    print("started dumping pickle file")

    with open('viz_bbox_data.pickle', 'wb') as handle:
        pickle.dump(bboxs_in_pcap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("finished dumping pickle file")



if __name__ == "__main__":
    start_process(r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\1.pcap", 
    r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\meta.json", 
    r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\1.csv")