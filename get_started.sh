#!/usr/bin/env bash
# Downloads raw data into ./download
# and saves preprocessed data into ./data
# Get directory containing this script

CODE_DIR="$( pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR

pip install -r $CODE_DIR/requirements.txt

# download punkt, perluniprops
if [ ! -d "/usr/local/share/nltk_data/tokenizers/punkt" ]; then
    python -m nltk.downloader punkt
fi


# SQuAD preprocess is in charge of downloading
# and formatting the data to be consumed later
DATA_DIR=data
DOWNLOAD_DIR=download
mkdir -p $DATA_DIR
rm -rf $DATA_DIR
python $CODE_DIR/run_preprocess.pt

# Data processing for TensorFlow
python $CODE_DIR/qa_data.py --glove_dim 100