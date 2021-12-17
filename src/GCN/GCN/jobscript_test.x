TEST_FILE="../data/test.txt"

# Test options
BATCH_SIZE=50
ST_EPOCH=1
EN_EPOCH=1

# Model options
HIDDEN_DIM=256
N_GCN_LAYER=4
N_PREDICTOR_LAYER=1
N_PROPERTIES=4
DROPOUT=0.1

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

date
