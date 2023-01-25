from collections import defaultdict, deque
import copy
import csv
import math
import ouster_utils
import typer
from pathlib import Path
import ast
import pickle


def process(pcap_path: Path = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,)
    ,meta_path: Path = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,)
    ,csv_path: Path = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,)):
    '''
    pcap_path: the pcap file

    meta_path: the metadata file of the pcap
    

    '''

    pcap_path = str(pcap_path)
    meta_path = str(meta_path)
    csv_path = str(csv_path)

    preds = defaultdict(list)
    with open(csv_path,"r") as f:
        for line in csv.DictReader(f):
            preds[int(line['frame'])].append(line)
            
    bboxs_in_pcap = dict()

    pcap = ouster_utils.Pcap(pcap_path,meta_path)
    print("started processing")
    for i,scan in enumerate(pcap):
        if len(preds[scan.frame_id]) != 0:
            print("\nPredctions in frame ", scan.frame_id)
            bboxs = []
            scan.process()
            for pred in preds[scan.frame_id]:
                x,y,w,h = ast.literal_eval(pred['xywh'])
                
                if math.isclose(w,0) or math.isclose(h,0):
                    continue
                cuboids,bbox2D = scan.process_pred(x,y,w,h)

                bboxs.extend(cuboids)
                bboxs.append(bbox2D)

            if bboxs:
                bboxs_in_pcap[scan.frame_id] = bboxs
            
            print("\tMoving on to the next frame")

    print("Finished processing")

    print("started dumping pickle file")

    with open('viz_bbox_data.pickle', 'wb') as handle:
        pickle.dump(bboxs_in_pcap, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("finished dumping pickle file")
    


if __name__ == "__main__":
    # typer.run(process)
    process(r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\1.pcap", 
    r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\meta.json", 
    r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section2\1.csv")