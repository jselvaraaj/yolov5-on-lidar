from os import listdir, rename
from os.path import join, isdir, abspath
from pathlib import Path
import subprocess
import shutil
import platform

import typer
from joblib import Parallel, delayed
from rich.progress import (
    SpinnerColumn,
    Progress,
    TextColumn,
)


def parallel_dir_break_up(base, break_up_fun):
    all_pcap_files = []
    all_meta_files = []
    all_save_paths = []

    def recurse_get_pcap_files(base, base_save_path=None):

        nonlocal all_pcap_files, all_meta_files, all_save_paths

        if base_save_path is None:
            base_save_path = abspath(base)

        files = listdir(base)

        directories = list(filter(lambda x: isdir(join(base, x)), files))

        pcap_files = list(map(lambda x: join(base, x), filter(lambda x: x.endswith(".pcap"), files)))

        meta_files = list(map(lambda x: join(base, x),filter(lambda x: x.endswith(".json"), files)))

        if len(meta_files) != 1:
            meta_files = map(lambda x: str(Path(x).with_suffix(".json").resolve()), pcap_files)
        else:
            meta_files = [meta_files[0]] * len(pcap_files)

        all_pcap_files.extend(pcap_files)
        all_meta_files.extend(meta_files)

        temp_save_paths = []

        for pcap in pcap_files:
            p = join(base_save_path, Path(pcap).stem)

            Path(p).mkdir(parents=True, exist_ok=True)

            temp_save_paths.append(p)

        all_save_paths.extend(temp_save_paths)

        for directory in directories:
            recurse_get_pcap_files(join(base, directory), join(base_save_path, directory))

    recurse_get_pcap_files(base)

    if all_pcap_files:
        Parallel(n_jobs=4, prefer="threads")(delayed(break_up_fun)(pcap, meta, save_path) for pcap, meta, save_path in
                                             zip(all_pcap_files, all_meta_files, all_save_paths))


progress_text = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=True,
)


def break_up_pcap(pcap, meta, save_path):
    print(f"Started {pcap}")
    shutil.copyfile(meta, join(save_path, "meta.json"))

    if platform.system() == "Windows":
        subprocess.run(["SplitCap.exe", "-r", pcap, "-o", save_path, "-s", "seconds", "360", "-b", "100000"])
    else:
        subprocess.run(["tcpdump", "-r", pcap, "-w", join(save_path, "small.pcap"), "-C", "5000"])

    files = listdir(save_path)
    files = list(filter(lambda x: x.endswith(".pcap"),files))

    files = sorted(files,key = lambda e : int(e[e.find("Seconds_")+8:-5]) )

    for i,e in enumerate(files):
        renamed = f"{i+1}.pcap"
        rename(join(save_path, e), join(save_path,renamed))


def split_from_dir(path: Path = typer.Option(
    ...,
    exists=True,
    file_okay=False,
    dir_okay=True,
    writable=True,
    readable=True,
    resolve_path=True, )):
    parallel_dir_break_up(str(path),break_up_pcap)


if __name__ == "__main__":
    typer.run(split_from_dir)
