from collections import defaultdict
import csv
import math
import os
from threading import Thread
import time
import ouster_utils
from ouster_viz import Visualizer
import typer
from pathlib import Path
import ast


def viz(pcap_path: Path = typer.Argument(
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
    resolve_path=True,)
    ,fps: float = typer.Option(10, help="Frames per second for the viz")):
    '''
    pcap_path: the pcap file

    meta_path: the metadata file of the pcap
    
    csv_path: the csv file containing the bounding boxes

    '''

    pcap_path = str(pcap_path)
    meta_path = str(meta_path)
    csv_path = str(csv_path)

    preds = defaultdict(list)
    with open(csv_path,"r") as f:
        for line in csv.DictReader(f):
            preds[int(line['frame'])].append(line)
            
    pcap = ouster_utils.Pcap(pcap_path,meta_path)

    def add_frames(viz,pcap, fps):

        for scan in pcap:
            if scan.frame_id in preds:
                
                for pred in preds[scan.frame_id]:
                    x,y,w,h = ast.literal_eval(pred['xywh'])

                    if math.isclose(w,0) or math.isclose(h,0):
                        continue

                    bbox = scan.process_pred(x,y,w,h)
                    viz.add_bbox(bbox)

            viz.add_scan(scan)
            
            viz.viewer.update()

            time.sleep(1/fps)
            viz.clear()

        


    with Visualizer() as viz:
        # add_frames(viz,pcap,fps)
        thread = Thread(target=add_frames, args=(viz,pcap,fps), daemon=True)

        thread.start()
        
        viz.run()


if __name__ == "__main__":
    typer.run(viz)