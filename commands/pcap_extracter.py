from typing import List

from ouster import client
from ouster import pcap
from contextlib import closing
from os import listdir
from os.path import join, isdir
from pathlib import Path
import typer

from enum import Enum
from rich.progress import (
    SpinnerColumn,
    Progress,
    TextColumn,
)

import cv2
import numpy as np
from joblib import Parallel, delayed

def transform_pcap(meta_path, file_path, save_path, clip, channels, extract_video, apply_filter):
    print(f"\n Working on {file_path}")

    with open(meta_path, 'r') as f:
        metadata = client.SensorInfo(f.read())

    pcap_file = pcap.Pcap(file_path, metadata)

    fps = int(str(metadata.mode)[-2:])
    width = int(str(metadata.mode)[:4])
    height = int(str(metadata.prod_line).split('-')[2])

    if clip:
        left_clip = 190  # 0
        right_clip = 275  # -1024
    else:
        left_clip = 0
        right_clip = -1024

    width -= left_clip + right_clip

    for channel in channels:
        Path(join(save_path, channel)).mkdir(parents=True, exist_ok=True)

    with closing(client.Scans(pcap_file)) as scans:

        if extract_video:
            vid_writer_sig = cv2.VideoWriter(save_path + "_signal.avi", cv2.VideoWriter_fourcc(*"RGBA"), fps,
                                             (width, height))
            vid_writer_range = cv2.VideoWriter(save_path + "_range.avi", cv2.VideoWriter_fourcc(*"RGBA"), fps,
                                               (width, height))
            vid_writer_ref = cv2.VideoWriter(save_path + "_reflectivity.avi", cv2.VideoWriter_fourcc(*"RGBA"), fps,
                                             (width, height))
            vid_writer_nir = cv2.VideoWriter(save_path + "_near_ir.avi", cv2.VideoWriter_fourcc(*"RGBA"), fps,
                                             (width, height))

        for i, scan in enumerate(scans):

            ref_field = scan.field(client.ChanField.REFLECTIVITY)
            ref_val = client.destagger(pcap_file.metadata, ref_field)
            ref_img = ref_val.astype(np.uint8)[:, left_clip:-right_clip]

            sig_field = scan.field(client.ChanField.SIGNAL)
            sig_val = client.destagger(pcap_file.metadata, sig_field)
            sig_img = sig_val.astype(np.uint8)[:, left_clip:-right_clip]

            ir_field = scan.field(client.ChanField.NEAR_IR)
            ir_val = client.destagger(pcap_file.metadata, ir_field)
            ir_img = ir_val.astype(np.uint8)[:, left_clip:-right_clip]

            range_field = scan.field(client.ChanField.RANGE)
            range_val = client.destagger(pcap_file.metadata, range_field)
            range_img = range_val.astype(np.uint8)[:, left_clip:-right_clip]

            custom_img = np.dstack((sig_img, range_img, ir_img))

            sig_img = np.dstack((sig_img, sig_img, sig_img))
            range_img = np.dstack((range_img, range_img, range_img))
            ref_img = np.dstack((ref_img, ref_img, ref_img))
            ir_img = np.dstack((ir_img, ir_img, ir_img))

            if apply_filter:
                # sig_img = cv2.cvtColor(sig_img, cv2.COLOR_BGR2HSV)
                sig_img = cv2.applyColorMap(sig_img, cv2.COLORMAP_TURBO)

            for channel in channels:
                name = join(save_path, channel, f"{i + 1}.png")
                if channel == "ir":
                    cv2.imwrite(name, ir_img)
                    if extract_video:
                        vid_writer_nir.write(ir_img)
                elif channel == "signal":
                    cv2.imwrite(name, sig_img)
                    if extract_video:
                        vid_writer_sig.write(sig_img)
                elif channel == "reflection":
                    cv2.imwrite(name, ref_img)
                    if extract_video:
                        vid_writer_ref.write(ref_img)
                elif channel == "range":
                    cv2.imwrite(name, range_img)
                    if extract_video:
                        vid_writer_range.write(range_img)
                else:
                    cv2.imwrite(name, custom_img)

        if extract_video:
            vid_writer_sig.release()
            vid_writer_range.release()
            vid_writer_ref.release()
            vid_writer_nir.release()

        cv2.destroyAllWindows()
        print(f"\nFinished {file_path}")


progress_text = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=True,
)


def recurse_transform(base, transform_fun, clip, channels, extract_video, apply_filter):
    with progress_text as progress:

        files = listdir(base)

        directories = list(filter(lambda x: isdir(join(base, x)), files))

        pcap_files = list(filter(lambda x: x.endswith(".pcap"), files))

        # meta_files = map(lambda x: str(Path(x).with_suffix(".json")),pcap_files)
        meta_files = list(map(lambda x: join(base, x),filter(lambda x: x.endswith(".json"), files)))

        if len(meta_files) != 1:
            meta_files = map(lambda x: str(Path(x).with_suffix(".json").resolve()), pcap_files)
        else:
            meta_files = [meta_files[0]] * len(pcap_files)

        if pcap_files:
            progress.add_task(description=f"Started a folder containing {pcap_files}", total=None)

            Parallel(n_jobs=4)(
                delayed(transform_fun)(meta, join(base, pcap), join(base, Path(pcap).stem), clip, channels,
                                       extract_video, apply_filter) for
                meta, pcap in
                zip(meta_files, pcap_files))

            progress.add_task(description=f"Finished {pcap_files}", total=None)

        for directory in directories:
            recurse_transform(join(base, directory), transform_fun, clip, channels, extract_video, apply_filter)


class Channel(str, Enum):
    signal = "signal"
    range = "range"
    ir = "ir"
    reflection = "reflection"


def extract_pcap_frames(base_dir: Path = typer.Option(
    ...,
    exists=True,
    file_okay=False,
    dir_okay=True,
    writable=True,
    readable=True,
    resolve_path=True, )
        , channels: List[Channel] = [Channel.signal]
        , clip: bool = False
        , extract_video: bool = False
        , apply_filter: bool = False):
    '''
    Extracts the frames from the pcap files in the subdirectories specified.

    base_dir: The root directory to start the search from

    channels: signal, ir, range, reflection

    clip: Clip the left and right side of the frames
    '''
    recurse_transform(str(base_dir), transform_pcap, clip, channels, extract_video, apply_filter)


if __name__ == "__main__":
    typer.run(extract_pcap_frames)
