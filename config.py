import argparse


def int_or_str(value):
    """Custom function to allow both int and str types."""
    try:
        return int(value)  # Try converting to integer
    except ValueError:
        return value  # If conversion fails, return as string


def MyParser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # general training
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--multinodes", type=int, default=0)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        help="we might need float32 for NAR model",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="per gpu")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--tb_write_every_n_steps", type=int, default=100)
    parser.add_argument("--print_every_n_steps", type=int, default=250)
    parser.add_argument("--val_every_n_steps", type=int, default=500)
    parser.add_argument(
        "--inference_every_n_steps",
        type=int,
        default=3000,
        help="will only get to inference when model is saved, and therefore this needs to be multiple of val_every_n_steps",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=10000000,
        help="save the model every n steps, will save the model as bundle_step$step.pth",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="this is the effective batch size per gpu, no matter whether using gradient_accumulation_steps",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.1,
        help="use linear warmup, the proportion of the training steps that are used for warming up",
    )
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="if not None, will ignore n_epochs and use num_steps as the total number of amount of training, can try e.g. 400000 i.e. 400k steps",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="the value for torch.nn.utils.clip_grad_norm_()",
    )
    parser.add_argument(
        "--early_stop_step",
        type=int,
        default=3200,
        help="stop training after this many steps of non-improvement",
    )
    parser.add_argument(
        "--early_stop_threshold",
        type=float,
        default=-1.0,
        help="early stop after the improvement is below this threshold for certain number of steps",
    )

    # path
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="/saltpool0/scratch/pyp/VoiceEditor/",
        help="will be combined with dataset name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="e.g. 'libritts', 'librilight', 'spotify', they are folder name in the data dir also",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="need to be compatible with corresponding dataset py file",
    )
    parser.add_argument(
        "--compact_folder_name",
        type=str,
        default=None,
        help="if not None, will use compact_combined_dataset.py, and this is the folder name of the compact dataset",
    )
    parser.add_argument(
        "--inference_dataset_dir",
        type=str,
        default="/data/scratch/pyp/datasets/librilight/preprocessed",
        help="need to be compatible with corresponding dataset py file",
    )

    parser.add_argument(
        "--training_stage",
        type=int,
        default=1,
        help="if 1, train VoiceEditor_one, if 2 train VoiceEditor_seven",
    )
    parser.add_argument(
        "--local_wandb",
        type=int,
        default=0,
        help="if 1, will use local wandb, otherwise use the global one",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="puyuanpeng",
        help="the entity (usually your username) for wandb",
    )
    # data
    parser.add_argument(
        "--librilight_ratio",
        type=float,
        default=1,
        help="the portion of lightlight compared to gigaspeech, 1 means equal, 2 means librilight data is twice as much as gigaspeech",
    )
    parser.add_argument(
        "--plus_librilight_root",
        type=str,
        default=None,
        help="if not None, will combine gigaspeech and librilight, this is the root folder to librilight. Note that will need to merge the vocab.txt based on gigaspeech's, in order to be able to load a pretrained model",
    )
    parser.add_argument(
        "--plus_librilight_phn_folder_name",
        type=str,
        default=None,
        help="if not None, will combine gigaspeech and librilight, this is the phoneme folder name of librilight",
    )
    parser.add_argument(
        "--plus_librilight_encodec_folder_name",
        type=str,
        default=None,
        help="if not None, will combine gigaspeech and librilight, this is the encodec folder name of librilight",
    )
    parser.add_argument(
        "--plus_librilight_manifest_name",
        type=str,
        default=None,
        help="if not None, will combine gigaspeech and librilight, this is the manifest folder name of librilight",
    )
    parser.add_argument(
        "--skip_us",
        type=int,
        default=0,
        help="skip the giga utterances that contains 'j uː ɛ s' because of the tokenization issue",
    )
    parser.add_argument(
        "--pseudo_epoch_size",
        type=int,
        default=37901,
        help="only use for Eden scheduler. 37901 is the epoch size in the default optim setting, this is probably too big",
    )
    parser.add_argument(
        "--switch_order",
        type=int,
        default=0,
        help="this is only for hificodec, where we switch the order of 2 and 3nd codebook",
    )
    parser.add_argument(
        "--phn_folder_name",
        type=str,
        default="phoneme",
        help="for libritts I also have arpa phns, in which case should be phonemes_arpa",
    )
    parser.add_argument(
        "--encodec_folder_name",
        type=str,
        default="mimi_8cb",
        help="folder where encodec codes are stored",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="manifest_final",
        help="if using hificodec, it should be hificodec_menifest, if using encodec, it is the default",
    )
    parser.add_argument(
        "--pad_x",
        type=int,
        default=1,
        help="whether or not always pad x to have text_max_length. select 1 to get the maximal memory consumption, but the actual case should be smaller, better to have it being 0",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=18750,
        help="max number of encodec tokens per gpu, this is only used when using dynamic batching, will ignore batch size. Note that batch size is the final effective batch size (sum of batch on each gpu), but max_num_tokens is per gpu",
    )
    parser.add_argument(
        "--val_max_num_tokens",
        type=int,
        default=6000,
        help="FOR validation, this basically is for music-gen because of high mem consumption. max number of encodec tokens per gpu, this is only used when using dynamic batching, will ignore batch size. Note that batch size is the final effective batch size (sum of batch on each gpu), but max_num_tokens is per gpu",
    )
    parser.add_argument("--num_buckets", type=int, default=10)
    parser.add_argument("--dynamic_batching", type=int, default=1)
    parser.add_argument(
        "--audio_max_length",
        type=float,
        default=120,
        help="in second, crop the audio is length is longer than this",
    )
    parser.add_argument(
        "--audio_min_length",
        type=float,
        default=2,
        help="in second, drop the audio if length is shorter than this",
    )
    parser.add_argument(
        "--text_max_length", type=int, default=1000, help="if too long, we crop"
    )
    parser.add_argument(
        "--text_min_length", type=float, default=10, help="if too short, will drop"
    )
    parser.add_argument(
        "--encodec_sr",
        type=float,
        default=50,
        help="for 24kHz mimi model, it produces 12.5 codes for 1 sec of audio",
    )
    parser.add_argument(
        "--mask_len_min", type=int, default=20, help="Minimum mask length"
    )
    parser.add_argument(
        "--mask_len_max", type=int, default=400, help="Maximum mask length"
    )
    parser.add_argument(
        "--extra_mask_len_min", type=int, default=2, help="Minimum extra mask length"
    )
    parser.add_argument(
        "--extra_mask_len_max", type=int, default=20, help="Maximum extra mask length"
    )
    parser.add_argument(
        "--final_audio_token_len",
        type=int,
        default=772,
        help="this is only for stage 1 training, since we add eog, start_of_continue, and a random amount of extra mask, --audio_max_length won't be the final max length, the self.args.final_audio_token_len = self.args.audio_max_length*self.args.encodec_sr+self.args.extra_mask_len_max+2 ",
    )

    # model
    parser.add_argument(
        "--ttsonly", default=0, type=int, help="if 1, only train tts model, no CM3"
    )
    parser.add_argument(
        "--load_existing_text_embedding",
        type=int,
        default=0,
        help="if 1, when load model and the text vocab doesn't match, will load the existing weights while the new weights will be initialized randomly",
    )
    parser.add_argument(
        "--fly", type=int, default=0, help="if 1, encode chunked audio on the fly"
    )
    parser.add_argument(
        "--encodec_ckpt",
        type=str,
        default="/data/scratch/pyp/exp_pyp/audiocraft/encodec/xps/6f79c6a8/checkpoint.th",
    )
    parser.add_argument(
        "--downsample_rate",
        type=int,
        default=320,
        help="the downsample rate for the encodec model, 16000/320 = 50Hz",
    )
    parser.add_argument(
        "--segtts_mask",
        type=int,
        default=0,
        help="if 1, use segtts_mask model, where we have a prefix and segment utterance into two and shifted separately for modeling, and use make use of mask:0, by insert two mask:0 in the middle of the two segments",
    )
    parser.add_argument(
        "--segtts",
        type=int,
        default=0,
        help="if 1, use segtts model, where we have a prefix and segment utterance into two and shifted separately for modeling",
    )
    parser.add_argument(
        "--edge",
        type=int,
        default=0,
        help="if 1, use edge prediction for the first codebook",
    )
    parser.add_argument(
        "--duration_loss_weight",
        type=float,
        default=1.0,
        help="weight on the duration loss",
    )
    parser.add_argument(
        "--drop_long",
        type=int,
        default=1,
        help="if this is true, will drop example whose encodec sequence or phone sequence is too long, rather than cropping as we did before, to avoid hellucination",
    )
    parser.add_argument(
        "--eos",
        type=int,
        default=2051,
        help="this is to be used with reduced_eog, where we end the utterance with eos, and end the generated segment with eog, also when this is used, the n_special should be 4",
    )
    parser.add_argument(
        "--reduced_eog",
        type=int,
        default=1,
        help="for the non-final segments, do not insert eog at the end, this could hopefully solve the early stopping issue when doing tts",
    )

    parser.add_argument(
        "--valle_orig",
        type=int,
        default=0,
        help="the original valle model, trained for TTS",
    )
    parser.add_argument("--valle_max_prompt_len", type=float, default=6, help="in sec.")
    # randomly choose a portion as tts examples during training
    parser.add_argument(
        "--tts_portion",
        type=float,
        default=0,
        help="randomly choose a portion of the training examples as tts examples, where no mask and rearrangement is used",
    )

    # put special tokens first to handle different vocab_size
    parser.add_argument(
        "--special_first",
        type=int,
        default=0,
        help="if 1, need to have special tokens to be the first few tokens, e.g. 0, 1, 2, which means we need to adjust the preprocessing and postprocessing of the encodec codes. note that we hard coded to have 3 special tokens",
    )
    parser.add_argument("--n_special", type=int, default=4, help="empty, eog, pad, eos")

    # weight codebook differently
    parser.add_argument(
        "--codebook_weight", type=str, default=None, help="e.g. ['5','1','0.5','0.1']"
    )

    # args for MusicGen
    parser.add_argument(
        "--mask_span_weight",
        default=1.0,
        type=float,
        help="the weight on the tokens in masked span",
    )
    parser.add_argument(
        "--unmask_span_weight",
        default=1.0,
        type=float,
        help="the weight on unmasked span",
    )
    parser.add_argument(
        "--start_end_weight",
        default=None,
        type=str,
        help="weight the start x tokens and end x tokens differently, e.g. (10,2.0), means x == 10, weight==2.0",
    )
    # for now not consider the two weights above, only consider eog_weight, which is defined below somewhere, as the above two are not super principled

    parser.add_argument(
        "--musicgen",
        type=int,
        default=0,
        help="whether or not use this model, will also have an impact on the output shape of the dataset",
    )
    parser.add_argument(
        "--enc_dec",
        default=0,
        type=int,
        help="use enc-dec architecture, text is from the enc, only for musicgen",
    )
    parser.add_argument(
        "--dec",
        default=0,
        type=int,
        help="use dec only architecture, text is from the enc, only for musicgen. Exclusive with --enc_dec",
    )
    parser.add_argument(
        "--empty_token",
        default=2048,
        type=int,
        help="indicating the no token at the position for the codebook",
    )
    # args for the optimizer and scheduler from Feiteng
    # original setup for the 3 params are 5000 4 and 1000
    # but that's because set_epoch is run on num_gradient_accumulation_step*step (with 4 being the accumulation step)
    # so I scaled down them a little bit
    # will try scaling them back if this doesn't work
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="AdamW",
        help="can also use ScaledAdam, in which case we'll also use the Eden scheduler",
    )
    parser.add_argument(
        "--reduce_lr_start_step",
        type=int,
        default=3000,
        help="after which significantly reduce the lr. a param for the eden optimizer",
    )
    parser.add_argument("--reduce_lr_start_epoch", type=int, default=4)
    parser.add_argument("--clipping_update_period", type=int, default=600)

    # below are args for valle
    # below are args for valle
    parser.add_argument(
        "--valle", type=int, default=0, help="if 1, use valle model (cm3)"
    )
    parser.add_argument("--decoder_dim", type=int, default=1024)
    parser.add_argument("--norm_first", action="store_true", default=True)
    parser.add_argument("--add_prenet", action="store_true", default=False)
    parser.add_argument(
        "--prefix_mode",
        type=int,
        default=5,
        help="this is for NAR, we only do 5, which is CM3",
    )
    parser.add_argument("--share_embedding", action="store_true", default=False)
    parser.add_argument("--nar_scale_factor", type=float, default=1.0)
    parser.add_argument("--prepend_bos", action="store_true", default=False)
    parser.add_argument(
        "--sync_nar",
        type=int,
        default=0,
        help="whether to choose the same NAR model to run for training_stage==2 across different process (this is only for DDP)",
    )
    # above are args for valle
    # above are args for valle

    # add parallel_pattern
    parser.add_argument(
        "--parallel_pattern",
        type=int,
        default=0,
        help="if 1, use parallel pattern, we also use LFSC codec",
    )
    parser.add_argument(
        "--full_prediction",
        type=int,
        default=0,
        help="this is for ve1, if 1, use full autoregressive mask, and calculate loss over all tokens, except for mask_tokens",
    )
    parser.add_argument(
        "--multicm3",
        type=int,
        default=0,
        help="cm3 model but allows multiple mask spans",
    )
    parser.add_argument(
        "--max_mask_portion",
        type=float,
        default=0.7,
        help="should mask a utterance for more than this portion",
    )
    parser.add_argument(
        "--max_n_spans",
        type=int,
        default=8,
        help="maximal number of spans, only use when using multicm3, this is used to decide number of mask_embedding, and max clamp value if use Poisson distribution, if use uniform distribution to sample number of spans if will be uniform(1,max_n_spans)",
    )
    parser.add_argument(
        "--shuffle_mask_embedding",
        type=int,
        default=0,
        help="whether shuffle the mask embedding, so that mask:0 is not the most well trained, default is not shuffling. The default has it's benefit, as it make sure that mask:0 always appear the first",
    )
    parser.add_argument(
        "--mask_sample_dist",
        type=str,
        default="uniform",
        help="uniform or poissonx, e.g. poisson1, meaning the parameter lambda is 1, it will most likely sample 1 masks",
    )
    parser.add_argument(
        "--min_gap",
        type=int,
        default=10,
        help="after sampled starts, delete later one if it closer to the former start than the min_gap",
    )

    parser.add_argument(
        "--cm3",
        type=int,
        default=0,
        help="use cm3 style for ve1, the input from dataloader is going to be just raw data, all masking and rearrangement will happen whin the model",
    )

    parser.add_argument(
        "--sep_special_token",
        type=int,
        default=0,
        help="remove text/audio pad token, set audio_mask_token and start of continue to be separately learned embeddings. Therefore, for ve1 self.n_text_tokens == self.args.text_vocab_size, self.n_audio_tokens == self.args.audio_vocab_size + 2, for ve7, self.n_text_tokens == self.args.text_vocab_size, self.n_audio_tokens == self.args.audio_vocab_size",
    )
    parser.add_argument(
        "--one_causal",
        type=int,
        default=0,
        help="whether model VE_one generation as autoregressive gen or non-autoregressive gen",
    )
    parser.add_argument("--n_codebooks", type=int, default=8)
    parser.add_argument(
        "--weight_sharing",
        type=int,
        default=0,
        help="sharing weights between VE_seven predict layer and embedding layer",
    )
    parser.add_argument(
        "--text_vocab_size", type=int, default=86, help="Size of text vocabulary"
    )
    parser.add_argument(
        "--text_pad_token",
        type=int,
        default=86,
        help="padding of the text tokens, not attended",
    )
    # parser.add_argument('--audio_vocab_size', type=int, default=1024, help='Size of audio vocabulary')
    parser.add_argument(
        "--audio_vocab_size",
        type=str,
        default="2048",
        help="Size of audio vocabulary, can be specified as '[128,512,1024,2048]'",
    )
    parser.add_argument(
        "--audio_mask_token",
        type=int,
        default=1024,
        help="Audio mask token, this the the extra mask used in the masked region for AR, for NAR, the entire masked region will be filled with it",
    )
    parser.add_argument(
        "--bog", type=int, default=1025, help="Begin of generation token"
    )
    parser.add_argument("--eog", type=int, default=2049, help="End of generation token")
    parser.add_argument(
        "--start_of_continue",
        type=int,
        default=1027,
        help="this token follows the masked region, proceeds the first unmasked token, to indicate that gt tokens starts",
    )
    parser.add_argument(
        "--audio_pad_token",
        type=int,
        default=2050,
        help="padding of the encodec codes, not attended",
    )
    parser.add_argument("--d_model", type=int, default=1024, help="Model dimension")
    parser.add_argument(
        "--audio_embedding_dim",
        type=int,
        default=128,
        help="dimension for encodec continues embedding (before being quantized)",
    )
    parser.add_argument(
        "--text_embedding_dropout",
        type=float,
        default=0.1,
        help="Dropout for text embedding",
    )
    parser.add_argument(
        "--audio_embedding_dropout",
        type=float,
        default=0,
        help="Dropout for audio embedding",
    )
    parser.add_argument(
        "--text_positional_embedding_dropout",
        type=float,
        default=0.1,
        help="Dropout for text positional embedding",
    )
    parser.add_argument(
        "--audio_positional_embedding_dropout",
        type=float,
        default=0.1,
        help="Dropout for audio positional embedding",
    )
    parser.add_argument(
        "--trm_dropout", type=float, default=0.1, help="Dropout for transformer"
    )
    parser.add_argument(
        "--nhead", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=12, help="Number of encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=12, help="Number of decoder layers"
    )
    parser.add_argument(
        "--eog_weight",
        type=float,
        default=1.0,
        help="Weight for End of generation token",
    )
    parser.add_argument(
        "--stage_one_load_encodec_embedding",
        type=str,
        default=None,
        help="Path to load encodec embedding for stage one. On our lab machine it is /saltpool0/scratch/pyp/VoiceEditor/encodec_embedding/24khz_8codebooks.pth, 8 is the n_codebooks",
    )
    parser.add_argument(
        "--stage_two_load_encodec_embedding",
        type=str,
        default=None,
        help="Path to load encodec embedding for stage two。 On our lab machine it is /saltpool0/scratch/pyp/VoiceEditor/encodec_embedding/24khz_8codebooks.pth, 8 is the n_codebooks",
    )
    parser.add_argument(
        "--stage_two_load_ve_one_embedding",
        type=str,
        default=None,
        help="Path to load VoiceEditor_one audio embedding for stage two",
    )
    parser.add_argument(
        "--load_model_from",
        type=str,
        default=None,
        help="Path to load model from, this will be effective last, so will overwrite all previous load, including resume",
    )
    parser.add_argument(
        "--load_model_from_ve1",
        type=str,
        default=None,
        help="Path to load ve1 model weights from, this will be effective last, designed for loading the encoder weights of the VE7 from a pretrained VE1",
    )

    ## below are args for the new long model
    parser.add_argument(
        "--target_time_stretch_prob",
        type=float,
        default=0,
        help="the probability of time stretching the target audio",
    )
    parser.add_argument(
        "--target_time_stretch_bound",
        type=float,
        default=0.1,
        help="the bound of the time stretching target audio, e.g. 0.1 means the audio will be stretched by 0.9 to 1.1",
    )
    parser.add_argument(
        "--time_stretch_prob",
        type=float,
        default=0,
        help="the probability of time stretching the audio",
    )
    parser.add_argument(
        "--time_stretch_bound",
        type=float,
        default=0.3,
        help="the bound of the time stretching, e.g. 0.3 means the audio will be stretched by 0.7 to 1.3",
    )
    parser.add_argument(
        "--no_loss_on_prefix",
        type=int,
        default=0,
        help="if 1, will not calculate loss on the prefix acoustic tokens",
    )
    parser.add_argument(
        "--x_sep_token",
        type=int,
        default=None,
        help="if not None, will use this token in between prompt text and target generation text",
    )
    parser.add_argument(
        "--y_sep_token",
        type=int,
        default=None,
        help="if not None, will use this token in between prompt codec tokens and target codec tokens",
    )
    parser.add_argument(
        "--neighbor_prompt_prob",
        type=float,
        default=0,
        help="the probability of using the prompt from the neighbor",
    )
    parser.add_argument(
        "--neighbor_folder_name",
        type=str,
        default="neighbors",
        help="folder where the neighbors of the current audio files are stored, each row contains three tab separated entries: neighbor_fn, neighbor_temporal_distance, neighbor_duration",
    )
    parser.add_argument(
        "--alignment_folder_name",
        type=str,
        default="alignment",
        help="folder where the forced alignment of the current audio files are stored, in csv format, each row contains five comma separated entries: begin, end, label, type, speaker, the first row is header",
    )
    parser.add_argument(
        "--ipa_alignment_folder_name",
        type=str,
        default="ipa_alignment",
        help="folder where the forced alignment of the current audio files are stored, in txt format, each row contains three tab separated entries: begin, end, ipa phn sequence, generated using data/ll60k_preprocessing/step7_ipa_alignment.py",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=float,
        default=30,
        help="in sec., maximal prompt length selected from some neighboring file",
    )
    parser.add_argument(
        "--min_prompt_len",
        type=float,
        default=0.5,
        help="in sec., minimal prompt length selected from some neighboring file",
    )
    parser.add_argument(
        "--neighbor_selection_method",
        type=str,
        default="maxdist_60",
        help="maxdist_60 means uniformly select a neighbor that's within 60 sec of the current audio file",
    )
    parser.add_argument(
        "--num_trial", type=int, default=5, help="number of tries to select a neighbor"
    )
    parser.add_argument(
        "--prompt_start_from_begining_prob",
        type=float,
        default=0.5,
        help="the probability of starting the prompt from the beginning of the neighbor",
    )
    parser.add_argument(
        "--min_alignment_len", type=int, default=5, help="in number of words"
    )
    parser.add_argument(
        "--audio_folder_name",
        type=str,
        default="audio",
        help="folder where the audio files are stored",
    )

    # rope parameters
    parser.add_argument(
        "--decoder_regular_rope",
        type=int,
        default=0,
        help="if 1, will use regular rope for the decoder (note that we always use regular rope for encoder). ",
    )
    parser.add_argument(
        "--progress_no_multiple",
        type=int,
        default=0,
        help="if 1, will not multiple the percentage progress by the length of the key, see apply_rotary_pos_emb in models/modules/activation.py, this applies to both rope and sinusoidal positional encoding. Note that progress scale is still applied, i.e. when we only apply progress scale, but not multiple, the scaling factor is constant for every sample, rather than sample dependent",
    )
    parser.add_argument(
        "--add_eos_to_text",
        type=int,
        default=0,
        help="if not 0, use this number as eos and add to the end of text token, usually use the second to last token in the vocab size",
    )
    parser.add_argument(
        "--add_bos_to_text",
        type=int,
        default=0,
        help="if not 0, use this number as bos and add to the begining of text token, usually use the third to last token in the vocab size",
    )
    parser.add_argument(
        "--use_sinusoidal",
        type=int,
        default=0,
        help="if 1, will use sinusoidal positional encoding, otherwise use rope. BUT if rope_base is None, will use sinusoidal",
    )
    parser.add_argument(
        "--sinusoidal_base",
        type=int,
        default=1e4,
        help="the base of the exponential function, default is 1e4",
    )
    parser.add_argument(
        "--use_sinusoidal_progress",
        type=int,
        default=0,
        help="if 1, will use sinusoidal positional encoding for progress, otherwise use rope",
    )
    parser.add_argument(
        "--rope_base",
        type=int,
        default=None,
        help="the base of the exponential function, default is 1e4, if None, will not use rope",
    )
    parser.add_argument(
        "--multiple_key_length",
        type=int,
        default=0,
        help="if 1, during progress calculation, will multiple the precentage progress by the length of the key, otherwise multiple with length of query. see models/rope_playground.ipynb",
    )
    parser.add_argument(
        "--progress_scale",
        type=float,
        default=1.0,
        help="scale the progress, the smaller the value, the bigger the diagonal in attention score, see models/rope_playground.ipynb",
    )

    # attention alignment loss
    parser.add_argument(
        "--attention_alignment_loss",
        type=float,
        default=0.0,
        help="the weight on the attention alignment loss, if 0, will not calculate the loss",
    )
    parser.add_argument(
        "--alignment_loss_layer",
        type=str,
        default="['0-1', '2', '3']",
        help='the layers to calculate the alignment loss, e.g. ["0-1", "2", "3"]',
    )
    parser.add_argument(
        "--alignment_loss_head",
        type=str,
        default="['0-1', '2', '3']",
        help='the attention heads to calculate the alignment loss, e.g. ["0-1", "2", "3"]',
    )
    parser.add_argument(
        "--alignment_blank_logit",
        type=float,
        default=-1.0,
        help="the logit for the blank token added to the attention weights",
    )

    # inference parameters
    parser.add_argument(
        "--metrics",
        type=str,
        default="['spk_sim','wer','mcd','pitch','energy','pesq','utmos']",
    )
    parser.add_argument(
        "--res_jsonl_root", type=str, default="/home/pyp/BoostedVoiceEditor/res"
    )
    parser.add_argument("--res_name", type=str, default="2jan25.jsonl")
    parser.add_argument("--inference_seed", type=int, default=1)
    parser.add_argument("--codec_audio_sr", type=int, default=16000)
    parser.add_argument("--codec_sr", type=float, default=50)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--silence_tokens", type=list, default=[])
    parser.add_argument("--kvcache", type=int, default=0)
    parser.add_argument("--stop_repetition", type=int, default=3)
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument(
        "--inference_manifest_fns",
        type=str,
        default="['/home/pyp/BoostedVoiceEditor/manifests/debug.jsonl']",
    )
    parser.add_argument("--use_gt_duration", type=int, default=1)
    parser.add_argument(
        "--save_root",
        type=str,
        default="/data/scratch/pyp/exp_pyp/BoostedVoiceEditor/gens",
    )
    parser.add_argument(
        "--encodec_signature",
        type=str,
        default="/data/scratch/pyp/exp_pyp/audiocraft/encodec/xps/6f79c6a8/checkpoint.th",
    )
    parser.add_argument(
        "--extra_cutoff",
        type=float,
        default=5,
        help="in rare cases where the model doesn't follow specified target duration (only happened in extrapolation cases), we will terminate generation once the extra duration exceeds this value",
    )
    parser.add_argument(
        "--duration_margin",
        type=float,
        default=0.04,
        help="used along with extra_cutoff, when extra_cutoff is used (i.e. model doesn't follow specified target_duration), we terminate the generate, and cut the results to target_duration + duration_margin",
    )
    # add repeat_prompt and asr_model_name
    parser.add_argument(
        "--repeat_prompt",
        type=int_or_str,
        default=0,
        help="if 1, will repeat the prompt for each segment",
    )
    parser.add_argument(
        "--asr_model_name",
        type=str,
        default="w2v2",
        help="the name of the asr model, if not None, will use the asr model to generate the prompt",
    )

    # depth transformer parameters
    parser.add_argument("--depth_dec_num_layers", type=int, default=0)
    parser.add_argument("--depth_dec_d_model", type=int, default=768)
    parser.add_argument("--depth_dec_nhead", type=int, default=12)
    parser.add_argument(
        "--moshi_depth",
        type=int,
        default=0,
        help="if 1, will use the same parameterization as moshi, i.e. temporal trm output will gets added to every transformed token embedding",
    )

    parser.add_argument(
        "--validation_sample_cap",
        type=int,
        default=None,
        help="cap the validation data to this number",
    )
    parser.add_argument(
        "--no_libri_in_training",
        type=int,
        default=None,
        help="if 1, will not use librilight in training, only use in validation",
    )
    parser.add_argument(
        "--uniform_weight_start_step",
        type=int,
        default=1e50,
        help="set all codebook weight to be uniform starting from this step",
    )

    return parser
