# Test options
NUM_WORKERS=4
BATCH_SIZE=50
ST_EPOCH=1
EN_EPOCH=1
TEST_FILE="./../data/test.txt"

# Model options
HIDDEN_DIM=512
N_RNN_LAYER=2
N_PREDICTOR_LAYER=1
N_PROPERTIES=4
DROPOUT=0.1

for i in `seq $ST_EPOCH $EN_EPOCH`
do
  python -u ../script/test.py \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --test_file $TEST_FILE \
  --restart_file './results/save_'$i'0.pt' \
  --hidden_dim $HIDDEN_DIM \
  --N_RNN_layer $N_RNN_LAYER \
  --N_predictor_layer $N_PREDICTOR_LAYER \
  --N_properties $N_PROPERTIES \
  --dropout $DROPOUT
done

date
