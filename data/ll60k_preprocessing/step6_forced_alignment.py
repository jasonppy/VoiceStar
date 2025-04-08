import os, sys
import subprocess, tqdm
from concurrent.futures import ThreadPoolExecutor


def align_folders(audio_root, subfolder, subsubfolder):
    # Construct output folder path
    file_root = os.path.dirname(audio_root)
    out_folder = f"{file_root}/alignment/{subfolder}/{subsubfolder}"

    # Create the output directory
    os.makedirs(out_folder, exist_ok=True)

    # Construct the MFA align command
    command = [
        "mfa",
        "align",
        "--single_speaker",
        "-j",
        "8",
        "--clean",
        f"{audio_root}/{subfolder}/{subsubfolder}",
        "english_us_arpa",
        "english_us_arpa",
        out_folder,
        "--beam",
        "50",
        "--retry_beam",
        "400",
        "--output_format",
        "csv",
    ]

    # Run the command
    subprocess.run(command, check=True)


def main(
    file_root="/data/scratch/pyp/datasets/librilight/librilight_example_preprocessed",
    max_parallel_jobs=10,
    max_spk=100,
    partition="1/10",
    n_workers=64,
):
    # Find all subfolder/subsubfolder combinations
    tasks = []
    audio_root = os.path.join(file_root, "audio")
    for subfolder in os.listdir(audio_root):
        subfolder_path = os.path.join(audio_root, subfolder)
        if os.path.isdir(subfolder_path):
            for subsubfolder in os.listdir(subfolder_path):
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
                if os.path.isdir(subsubfolder_path):
                    tasks.append((audio_root, subfolder, subsubfolder))
    speaker_folder_map = {}
    for audio_root, subfolder, subsubfolder in tasks:
        if os.path.join(audio_root, subfolder) not in speaker_folder_map:
            speaker_folder_map[os.path.join(audio_root, subfolder)] = [
                os.path.join(audio_root, subfolder, subsubfolder)
            ]
        else:
            speaker_folder_map[os.path.join(audio_root, subfolder)].append(
                os.path.join(audio_root, subfolder, subsubfolder)
            )
    speaker_folder_partitions = []
    for audio_root_subfolder, speaker_folders in speaker_folder_map.items():
        speaker_folder_partitions.extend(
            [
                speaker_folders[i : i + max_spk]
                for i in range(0, len(speaker_folders), max_spk)
            ]
        )
    s, e = partition.split("/")
    s, e = int(s) - 1, int(e)
    cur_tasks = speaker_folder_partitions[s::e]
    import secrets, string
    import soundfile, glob
    from joblib import Parallel, delayed

    def delete_corrupted(fn):
        try:
            x = soundfile.read(fn)
        except:
            print(f"removing corrupted file: {fn}")
            os.remove(fn)

    for j, task in enumerate(tqdm.tqdm(cur_tasks)):
        # get subfolder for the current task
        subs = [item.split("/")[-2] for item in task]
        # assert that all subs are the same
        assert len(set(subs)) == 1, subs
        sub = subs[0]
        # randomly generate a foldername
        # generate a random character
        # make softlink from item in task to temp folder
        random_string = "".join(
            secrets.choice(string.ascii_letters + string.digits) for i in range(10)
        )
        temp_folder = os.path.join(file_root, "softlink_audio", random_string)
        os.makedirs(temp_folder, exist_ok=True)
        out_folder = f"{file_root}/alignment/{sub}"
        all_out_speaker_folders = [
            os.path.join(out_folder, os.path.basename(item)) for item in task
        ]
        if sum(os.path.isdir(curpath) for curpath in all_out_speaker_folders) == len(
            all_out_speaker_folders
        ):
            continue
        # remove audio files that are corrupted
        all_audio_files = [
            audiofile for item in task for audiofile in glob.glob(item + "/*/*.flac")
        ]
        Parallel(n_jobs=n_workers)(
            delayed(delete_corrupted)(audiofn) for audiofn in all_audio_files
        )
        for item in task:
            # make softlink from subsubfolder to a new folder in temp folder
            os.symlink(item, os.path.join(temp_folder, os.path.basename(item)))
        # run mfa on the linked folder, but save alignment to the correct folder
        command = f"mfa align -j {n_workers} {temp_folder} english_us_arpa english_us_arpa {out_folder} --beam 50 --retry_beam 200 --output_format csv --quiet --use_mp --temporary_directory {temp_folder}_temp"
        os.system(command)
        # delete the temp_folder
        os.system(f"rm -r {temp_folder}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
