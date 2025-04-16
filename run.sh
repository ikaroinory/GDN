clear

python main.py \
  --seed 5 \
  --dataset swat \
  --dtype float \
  --device cuda \
  --batch_size 32 \
  --epoch 30 \
  --slide_window 5 \
  --slide_stride 1 \
  --k 5 \
  --d_hidden 64 \
  --d_output_hidden 128 \
  --num_output_layer 1 \
  --lr 0.001 \
  --early_stop 20
