# Train options
NUM_WORKERS=4
BATCH_SIZE=50
EPOCH=500
LR=1e-4
LR_DECAY=1
SAVE_PER_EPOCH=10

# Model options
HIDDEN_DIM=256
N_GCN_LAYER=4
N_PREDICTOR_LAYER=1
N_PROPERTIES=4
DROPOUT=0.1

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
