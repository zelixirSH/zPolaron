#!/bin/bash


i=1

# ============================================
# Mode 1: Train from scratch
# ============================================
#echo "$(date): Training from scratch (i=${i})..."
#/media/data/wzh/env/bin/python \
#   train.py \
#   --train_lmdb_dir ./lmdb_data \
#    --valid_lmdb ./lmdb_data/valid.lmdb \
#    -epochs 500 \
#    -log log_${i}.dat \
#    -bestmodel best_model_${i}.pth \
#    -finalmodel final_model_${i}.pth \
#    -save_interval 5 \
#    -save_dir models_${i} \
#    -device cuda:0 \
#    -weight_decay 0.01 \
#    -patience 40 \
#    -delta 0.001 \
#    -batchsz 12 \
#    -lr 0.002 \
#    -dropout 0.2 \
#    -edge_loss_weight 8 \
#    -label_smoothing 0.1 \
#    -multi_gpu \
#    --num_workers 12 \
#    --accum_steps 2
##    -basemodel best_model_1.pth \

# ============================================
# Mode 2: Resume training
# Resume from checkpoint_latest.pth
# -epochs 500 stays, automatically runs from completed epoch+1 to 499
# ============================================
echo "$(date): Resume training (i=${i}), from checkpoint_latest.pth..."
/media/data/wzh/env/bin/python \
    train.py \
    -resume models_${i}/checkpoint_latest.pth \
    --train_lmdb_dir ./lmdb_data \
    --valid_lmdb ./lmdb_data/valid.lmdb \
    -epochs 500 \
    -log log_${i}.dat \
    -bestmodel best_model_${i}.pth \
    -finalmodel final_model_${i}.pth \
    -save_interval 5 \
    -save_dir models_${i} \
    -device cuda:0 \
    -weight_decay 0.01 \
    -patience 40 \
    -delta 0.001 \
    -batchsz 12 \
    -lr 0.002 \
    -dropout 0.2 \
    -edge_loss_weight 8 \
    -label_smoothing 0.1 \
    -multi_gpu \
    --num_workers 12 \
    --accum_steps 2
