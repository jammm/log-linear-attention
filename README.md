# log-linear-attention

# Setup

```bash
pip install -e .
pip install -e flame/
pip install -r flame/3rdparty/torchtitan/requirements.txt
```

# Data Preparation
Modify the save path in `hattention/preprocess_data.py`, then run the following command to preprocess the data:
```bash
python -m hattentions.preprocess_data
```

# Train

First, step into the training framework `flame/`
```bash
cd flame/
```

Then launch training!
```bash
bash ../scripts/train_flame.sh --name [NAME] --config [CONFIG] --seed [--ac]
```
The optional `--ac` flag is used to enable activation checkpointing. To specify `CONFIG`, use the name of the config file in `configs/flame/` but without the `.json` extension. Note that you will need to modify the absolute file-path in `scripts/train_flame.sh`.