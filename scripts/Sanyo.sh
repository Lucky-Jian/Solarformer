# ALL scripts in this file come from Autoformer
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

for model_name in   GRU  LSTM  TCN  Autoformer Transformer  Informer DLinear  Pyraformer Solarformer
do
for pred_len in 96 192
do
  python -u run.py \
    --itr 1 \
    --n_heads 8 \
    --d_model 512 \
    --dropout 0.05 \
    --learning_rate 0.01 \
    --root_path ./data/ \
    --data_path Sanyo.csv \
    --model_id Sanyo_$pred_len \
    --model $model_name \
    --data custom \
    --features MS \
    --target Power \
    --freq t \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 64 \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 1 \
    --patience 3 \
    --num_workers 0 \
    --train_epochs 10 >logs/LongForecasting/$model_name'_Sanyo_'$pred_len.log
done
done


