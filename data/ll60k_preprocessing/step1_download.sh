# define where to store the the downloaded data
dataroot=$DATAROOT
mkdir -p $dataroot
manifestroot=$dataroot/libriheavy
mkdir -p $manifestroot
audioroot=$dataroot/audio
mkdir -p $audioroot

# download libriheavy_long and libriheavy
cd $manifestroot
wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_dev.jsonl.gz?download=true -O libriheavy_cuts_dev.jsonl.gz
wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_test_clean.jsonl.gz?download=true -O libriheavy_cuts_test_clean.jsonl.gz
wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_test_other.jsonl.gz?download=true -O libriheavy_cuts_test_other.jsonl.gz
wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_small.jsonl.gz?download=true -O libriheavy_cuts_small.jsonl.gz
wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_medium.jsonl.gz?download=true -O libriheavy_cuts_medium.jsonl.gz
wget https://huggingface.co/datasets/pkufool/libriheavy/resolve/main/libriheavy_cuts_large.jsonl.gz?download=true -O libriheavy_cuts_large.jsonl.gz
wget https://huggingface.co/datasets/pkufool/libriheavy_long/resolve/main/libriheavy_cuts_small.jsonl.gz?download=true -O libriheavy_long_original_cuts_small.jsonl.gz
wget https://huggingface.co/datasets/pkufool/libriheavy_long/resolve/main/libriheavy_cuts_medium.jsonl.gz?download=true -O libriheavy_long_original_cuts_medium.jsonl.gz
wget https://huggingface.co/datasets/pkufool/libriheavy_long/resolve/main/libriheavy_cuts_large.jsonl.gz?download=true -O libriheavy_long_original_cuts_large.jsonl.gz

# turn .jsonl.gz to .jsonl
gunzip -k libriheavy_cuts_dev.jsonl.gz
gunzip -k libriheavy_cuts_test_clean.jsonl.gz
gunzip -k libriheavy_cuts_test_other.jsonl.gz
gunzip -k libriheavy_cuts_small.jsonl.gz
gunzip -k libriheavy_cuts_medium.jsonl.gz
gunzip -k libriheavy_cuts_large.jsonl.gz
gunzip -k libriheavy_long_original_cuts_small.jsonl.gz
gunzip -k libriheavy_long_original_cuts_medium.jsonl.gz
gunzip -k libriheavy_long_original_cuts_large.jsonl.gz

# if librilight is already unzipped in origDATAROOT, then skip this step
# download ll
cd $audioroot
wget https://dl.fbaipublicfiles.com/librilight/data/small.tar
wget https://dl.fbaipublicfiles.com/librilight/data/medium.tar
wget https://dl.fbaipublicfiles.com/librilight/data/large.tar

# untar small, medium, large
tar -xf small.tar
tar -xf medium.tar
tar -xf large.tar