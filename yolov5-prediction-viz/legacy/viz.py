from collections import defaultdict, deque
import copy
import csv
import math
import os
import pickle
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
    ,fps: float = typer.Option(10, help="Frames per second for the viz")):
    '''
    pcap_path: the pcap file

    meta_path: the metadata file of the pcap
    
    csv_path: the csv file containing the bounding boxes

    '''

    pcap_path = str(pcap_path)
    meta_path = str(meta_path)
    viz_data_path = "viz_data.pickle"

    bboxs_in_pcap = {}
    coords = []

    with open(viz_data_path, 'rb') as handle:
        bboxs_in_pcap = pickle.load(handle)

    print(len(list(bboxs_in_pcap.values())[0]))
    
    def add_frames(viz,fps):
        nonlocal pcap_path,meta_path,bboxs_in_pcap

        for scan in ouster_utils.Pcap(pcap_path,meta_path)(buffer=True):
            viz.clear()

            if scan.frame_id in bboxs_in_pcap:
                three_d_bboxs,two_d_coord = bboxs_in_pcap[scan.frame_id]
                viz.add_scan(scan,two_d_coord)
                print("Adding bboxs to frame ", scan.frame_id)    
                for bbox in three_d_bboxs:
                    viz.add_bbox(bbox)
            else:
                viz.add_scan(scan)

            viz.viewer.update()
            
            time.sleep(1/fps)          


    with Visualizer() as viz:
        # add_frames(viz,pcap,fps)
        thread = Thread(target=add_frames, args=(viz,fps), daemon=True)

        thread.start()
        
        viz.run()


if __name__ == "__main__":
    # typer.run(viz)
    viz(r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section1\1.pcap"
    , r"..\..\data\from_car\OS1-64_2022-11-16\processed\Section1\meta.json"
    ,10)