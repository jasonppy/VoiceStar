import argparse
from email.policy import default
def parse_args():
    parser = argparse.ArgumentParser(description="encode the dataset using codec model")
    parser.add_argument('--root', type=str, default="/data/scratch/pyp/datasets/emilia", help="Path to the directory")
    parser.add_argument('--sub_root', type=str, default="preprocessed", help="sub directory")
    parser.add_argument('--encodec_name', type=str, default="encodec_6f79c6a8.th", help="name of the codec model")
    parser.add_argument('--n_workers', type=int, default=16, help="Number of parallel worker processes")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size for codec encoding, decrease it if OOM. This is the sum of batch size *over each gpu*, so increase it if you are using more gpus")
    parser.add_argument('--audio_sr', type=int, default=16000, help='input audio sample rate')
    parser.add_argument('--model_sr', type=int, default=16000, help='encodec input audio sample rate')
    parser.add_argument('--downsample_rate', type=int, default=320, help='encodec downsample rate')
    parser.add_argument('--model_code_sr', type=float, default=50, help='codec model code sample rate')
    parser.add_argument('--len_cap', type=float, default=1000, help='will drop audios that are longer than this number')
    parser.add_argument('--min_len', type=float, default=0.5, help='will drop audios that are shorter than this number')
    parser.add_argument('--partition', type=str, default="1/1", help='split for parallel processing')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'])
    return parser.parse_args()

if __name__ == "__main__":
    import logging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)

    import os, sys
    import numpy as np
    import torch
    import torchaudio
    import tqdm
    import time

    args = parse_args()

    def sort_by_audio_len(lens):
        inds = np.argsort(lens).tolist()
        
        logging.info(f"longest: {lens[inds[-1]]/args.downsample_rate} encodec codes, {lens[inds[-1]]/args.model_sr:.2f} sec.")
        logging.info(f"shortest: {lens[inds[0]]/args.downsample_rate} encodec codes, {lens[inds[0]]/args.model_sr:.2f} sec.")
        logging.info(f"median: {lens[inds[len(inds)//2]]/args.downsample_rate} encodec codes, {lens[inds[len(inds)//2]]/args.model_sr:.2f} sec.")
        logging.info(f"95 percentile longest: {lens[inds[int(len(inds)*0.95)]]/args.downsample_rate} encodec codes, {lens[inds[int(len(inds)*0.95)]]/args.model_sr:.2f} sec.")
        return inds[::-1]

    def write_array_to_txt_file(array, filename):
        with open(filename, 'w') as f:
            for a in array[:-1]:
                f.write(' '.join(map(str, a))+'\n')
            f.write(' '.join(map(str, array[-1])))

    class mydataset(torch.utils.data.Dataset):
        def __init__(self, split):
            super().__init__()
            self.split = split
            self.audio_dir = audio_dir
            manifest_fn = os.path.join(encodec_manifest_dir, split+".txt")
            cur_sp = int(args.partition.split("/")[0])-1
            total_sp = int(args.partition.split("/")[1])
            with open(manifest_fn, "r") as rf:
                self.data = [l.strip().split("\t") for l in rf.readlines()][cur_sp::total_sp]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, ind):
            try:
                afn = self.data[ind][0]
                fn = os.path.join(self.audio_dir, afn)
                audio, sr = torchaudio.load(fn)
                if sr != args.model_sr:
                    audio = torchaudio.transforms.Resample(sr, args.model_sr)(audio)
                    sr = args.model_sr
                assert sr == args.model_sr, sr
            except Exception as e:
                # logging.info(f"{e}")
                return None, None, None
            assert audio.ndim==2 and audio.shape[0] == 1, audio.shape
            return audio.type(torch.float32).squeeze(0), audio.shape[-1], os.path.splitext(afn)[0]
        def collate(self, batch):
            lens, audios, segment_ids = [], [], []
            for item in batch:
                if item[0] != None:
                    audios.append(item[0])
                    lens.append(item[1])
                    segment_ids.append(item[2])
            return audios, lens, segment_ids
    
    # roots
    sub_root = args.sub_root
    encodec_manifest_dir = os.path.join(args.root, sub_root, "manifest_for_codec")
    audio_dir = os.path.join(args.root, sub_root, "audio")
    save_manifest_dir = os.path.join(args.root, sub_root,"manifest_final_encodec")
    if args.encodec_name == "encodec_6f79c6a8.th":
        save_codes_dir = os.path.join(args.root, sub_root,"encodec_4cb")
    elif args.encodec_name == "encodec_8cb1024_giga.th":
        save_codes_dir = os.path.join(args.root, sub_root,"encodec_8cb")

    os.makedirs(save_manifest_dir, exist_ok=True)
    os.makedirs(save_codes_dir, exist_ok=True)
    
    def import_encodec():
        from encodec import get_compression_model
        userdir = os.path.expanduser("~")
        model = get_compression_model(os.path.join(userdir, "VoiceStar", f"pretrained/{args.encodec_name}"), encode_only=True, device="cuda")
        model = torch.nn.DataParallel(model)
        return model
    model = import_encodec()
    
    # setup dataloader
    mega_batch_size = 2048
    batch_size = args.batch_size
    
    dataset = mydataset(args.split)
    if len(dataset) == 0:
        logging.info(f"no data found for split {args.split} partition {args.partition}")
        sys.exit(0)
    loader = torch.torch.utils.data.DataLoader(dataset, batch_size=mega_batch_size, shuffle=False, drop_last=False, num_workers=args.n_workers, collate_fn=dataset.collate)
    split = args.split

    skip = 0
    logging.info(f"now processing split {split} partition {args.partition}...")
    mega_n_steps = int(np.ceil(len(loader.dataset) / mega_batch_size))
    # mega_n_steps = int(np.ceil(len(gs) / mega_batch_size))
    logging.info(f"partition the split {split} into {mega_n_steps} parts, each has at most {mega_batch_size} samples")
    mani_fn = os.path.join(save_manifest_dir, f"{split}_{args.partition.replace('/', '=')}.txt")
    logging.info(f"manifest for split {split} partition {args.partition.replace('/', '=')}.txt will be saved at {mani_fn}")
    with open(mani_fn, "w") as mani_wf:
    # with open(mani_fn, "a") as mani_wf: # resume from where we failed
        for m, mega_batch in enumerate(tqdm.tqdm(loader, mininterval=60, maxinterval=60)):

            logging.info(f"====================================")
            logging.info(f"====================================")
            logging.info(f"now processing mega step {m+1}/{mega_n_steps}")

            try:
                lengths = np.array(mega_batch[1])
                sorted_inds = sort_by_audio_len(lengths)
                for j in range(len(sorted_inds))[::-1]:
                    if lengths[sorted_inds[j]] < args.model_sr*args.min_len or lengths[sorted_inds[j]] > args.model_sr*args.len_cap: # skip samples that are too short (shorter than 0.2s), or too big (bigger than 80s)
                        skip += 1
                        del sorted_inds[j]
                
                n_steps = int(np.ceil(len(sorted_inds) / batch_size))
                for n in tqdm.tqdm(range(n_steps), disable=True):
                    inds_used = sorted_inds[n*batch_size:(n+1)*batch_size]
                    wav_batch = [mega_batch[0][id] for id in inds_used]
                    all_lens = [mega_batch[1][id] for id in inds_used]
                    segment_id_batch = [mega_batch[2][id] for id in inds_used]
                    padded_wav = torch.nn.utils.rnn.pad_sequence(wav_batch, batch_first=True).unsqueeze(1) # [B, T] -> [B, 1, T]
                    # Extract discrete codes from EnCodec
                    with torch.no_grad():
                        if max(all_lens) > 300000 and len(all_lens) > 1: # if utterances are long, simply pass half of them at a time
                            codes = []
                            inwav = padded_wav.cuda()
                            codes.append(model(inwav[:len(inwav)//2])[0].cpu())
                            codes.append(model(inwav[len(inwav)//2:])[0].cpu())
                            codes = torch.cat(codes, dim=0)
                        else:
                            encoded_frames = model(padded_wav.cuda()) 
                            codes = encoded_frames[0].cpu() # [B, n_codebook, T]

                    for i, length in enumerate(all_lens):
                        save_fn = os.path.join(save_codes_dir, segment_id_batch[i]+".txt")
                        actual_len = round(length / args.downsample_rate) # 320 is downsample rate for this model
                        cur_code = codes[i].tolist() if type(codes) == list else codes[i, :, :actual_len].tolist()
                        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
                        write_array_to_txt_file(cur_code, save_fn)

                        mani_wf.write(f"{segment_id_batch[i]}\t{len(cur_code[0])}\n") # write to manifest file
                        # if i == 10:
                        #    raise
            except Exception as e:
                print(f'exception!! at {m+1}')
                print(e)
                continue

            # break
    logging.info(f"split {split} partition {args.partition} has {len(loader.dataset)} samples in total, skipped {skip} due to utterance being too long or too short")
        # break
