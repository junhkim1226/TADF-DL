# TADF-DL

github of Effect of molecular representation on deep learning performance for prediction of molecular electronic properties by Jun Hyeong Kim, Hyeonsu Kim, Woo Youn Kim.

## Table of Contents

- [Data](#data)
- [Train](#train)
- [Test](#test)

## Environmental

- python>=3.7
- [PyTorch]((https://pytorch.org/))=1.7.1
- [RDKit](https://www.rdkit.org/docs/Install.html)>=2020.09.3

## Data
### Extract molecules from PubChem

Move to `extract_data/`. Run `extract.py`, then run `choich.py` to get aromatic ring dataset.

### Data Structure

Basic data structure is shown below
```
ID    SMILES    HOMO    LUMO    E(S1)   E(T1)

```
For example,
```
id1   c1ccccc1    -5.6    -1.6    2.6   2.4
id2   Cc1ccccc1    -5.5    -1.5    2.7   2.3
...
```
Data file should be in `src/appropriate_model_directory/data/.`
Then, run `dataset_divide.py` to split dataset into train set and validation set, test set.

## Training

There are 4 models for predict TADF-related properties in `src/`.
Move to `src/appropriate_model_directory/train/`(e.g. `src/GCN/train/`)

Run `jobsctript_train.x`

Train script is shown below.
```shell
python -u ../script/train.py \
--num_workers $NUM_WORKERS \
--batch_size $BATCH_SIZE \
--num_epochs $EPOCH \
--lr $LR \
--lr_decay $LR_DECAY \
--save_every $SAVE_PER_EPOCH \
--hidden_dim $HIDDEN_DIM \
--N_GCN_layer $N_GCN_LAYER \
--N_predictor_layer $N_PREDICTOR_LAYER \
--N_properties $N_PROPERTIES \
--dropout $DROPOUT 1> ./results/log.txt
```
## Test

Run `jobsctript_test.x`

Test script is shown below.
```shell
for i in `seq $ST_EPOCH $EN_EPOCH`
do
  python -u ../script/test.py \
  --batch_size $BATCH_SIZE \
  --test_file $TEST_FILE \
  --restart_file './results/save_'$i'0.pt' \
  --hidden_dim $HIDDEN_DIM \
  --N_GCN_layer $N_GCN_LAYER \
  --N_predictor_layer $N_PREDICTOR_LAYER \
  --N_properties $N_PROPERTIES \
  --dropout $DROPOUT
done
```



