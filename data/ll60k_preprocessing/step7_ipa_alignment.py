# we have raw transcript at
# /data/scratch/pyp/datasets/librilight/preprocessed/audio
# we have word and ARPA alignment at
# /data/scratch/pyp/datasets/librilight/preprocessed/alignment

# we have manifest at /data/scratch/pyp/datasets/librilight/preprocessed/manifest_mimi
# where each row is like large/10022/essayoncriticism_1505_librivox_64kb_mp3/essayoncriticism_01_pope_64kb_5_610.32_630.08.flac	19.76

# we want to create IPA alignment from the raw transcript and word alignment, using phonemizer
# save at /data/scratch/pyp/datasets/librilight/preprocessed/ipa_alignment

# since ipa phonemized results are 1-to-1 with words (10 words might lead to a ipa sequence of 7 phonemes), we have to run phonemizer on each segment of the word sequence
import os, string, csv, random, tqdm, glob
from tokenizer import TextTokenizer, tokenize_text


def remove_punctuation(input_string):
    translator = str.maketrans("", "", string.punctuation)
    return input_string.translate(translator)


def create_alignment(
    fn,
    trans_dir,
    align_dir,
    audio_ext,
    trans_ext,
    arpa_ext,
    text_tokenizer,
    use_prob,
    ipa_alignment_fn,
    save=False,
    prompt_dur=30,
):
    os.makedirs(os.path.dirname(ipa_alignment_fn), exist_ok=True)
    trans_fn = os.path.join(trans_dir, fn.replace(audio_ext, trans_ext))
    if not os.path.isfile(trans_fn):
        return [], True
    align_fn = os.path.join(align_dir, fn.replace(audio_ext, arpa_ext))
    if not os.path.isfile(align_fn):
        return [], True
    # get raw transcript
    with open(trans_fn, "r") as f:
        transcript = f.read().strip()
    raw_word_list = transcript.split(" ")
    # get word alignment
    with open(align_fn, "r") as f:
        word_alignment = csv.reader(f)
        word_alignment = [row for row in word_alignment if row[3] == "words"]

    ipa_alignment = []

    for j, (item, raw_word) in enumerate(zip(word_alignment, raw_word_list)):
        start, end, word = float(item[0]), float(item[1]), item[2]
        if end > prompt_dur:
            break
        punc_re_raw_word = remove_punctuation(raw_word)
        if not remove_punctuation(word).lower() == punc_re_raw_word.lower():
            # print(f"word from alignment csv: {word}, word from txt: {raw_word}")
            return ipa_alignment, True
        if random.random() < use_prob:
            cur_words = " ".join(raw_word_list[: j + 1])
            phn = tokenize_text(text_tokenizer, cur_words)
            if len(phn) == 0:
                continue
            phn = " ".join(phn)
            start = (
                0  # at this point, we always start from the beginning of the sentence
            )
            ipa_alignment.append([start, end, phn])
    if save:
        if ipa_alignment:
            with open(ipa_alignment_fn, "w") as f:
                for item in ipa_alignment:
                    f.write(f"{item[0]}\t{item[1]}\t{item[2]}\n")
    else:
        return ipa_alignment, False


def main(
    data_root: str = "/data/scratch/pyp/datasets/librilight/preprocessed",
    audio_ext: str = ".flac",
    arpa_ext: str = ".csv",
    trans_ext: str = ".txt",
    split: str = "valid",
    use_prob: float = 0.5,
    max_dur: float = 30.0,  # do not consider utterance longer than this
    prompt_dur: float = 30.0,  # do not consider prompt longer than this
):
    text_tokenizer = TextTokenizer()
    trans_dir = f"{data_root}/audio"
    align_dir = f"{data_root}/alignment"
    manifest_fn = f"{data_root}/manifest_final_encodec/{split}*=*.txt"
    manifest_fns = glob.glob(manifest_fn)
    target_dir = f"{data_root}/ipa_alignment"
    encodec_sr = 50
    os.makedirs(target_dir, exist_ok=True)
    manifest = []
    for manifest_fn in manifest_fns:
        with open(manifest_fn, "r") as f:
            temp = [l.strip().split("\t") for l in f.readlines()]
            manifest += [
                l[0] + audio_ext for l in temp if float(l[1]) / encodec_sr < max_dur
            ]
    # # sequential processing
    n_flags = 0
    zero_words = 0
    for j, fn in enumerate(tqdm.tqdm(manifest)):
        ipa_alignment_fn = os.path.join(target_dir, fn.replace(audio_ext, ".txt"))
        ipa_alignment, flag = create_alignment(
            fn,
            trans_dir,
            align_dir,
            audio_ext,
            trans_ext,
            arpa_ext,
            text_tokenizer,
            use_prob,
            ipa_alignment_fn,
            prompt_dur=prompt_dur,
        )
        n_flags += flag
        if not ipa_alignment:
            zero_words += 1
        # print(f"{n_flags} out of {j+1} utterances have mismatched words")
        # print(f"{zero_words} out of {j+1} utterances have zero words")
        if ipa_alignment:
            with open(ipa_alignment_fn, "w") as f:
                for item in ipa_alignment:
                    f.write(f"{item[0]}\t{item[1]}\t{item[2]}\n")

    # # # # do the above using joblib parallisim
    # print(f"Processing {len(manifest)} utterances")
    # from joblib import Parallel, delayed
    # Parallel(n_jobs=32, verbose=2)(delayed(create_alignment)(fn, trans_dir, align_dir, audio_ext, trans_ext, arpa_ext, text_tokenizer, use_prob, os.path.join(target_dir, fn.replace(audio_ext, '.txt')), save=True) for fn in manifest)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
