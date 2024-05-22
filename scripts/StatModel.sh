# ARIMA and SARIMA is extremely slow, you might need to sample 1% data (add --sample 0.01)
# Naive is the Closest Repeat (Repeat-C). It repeats the last value in the look back window.

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

 for model_name in Naive GBRT  SVR ARIMA SARIMA

  do
  for pred_len in 96 192
    do
      python -u run_stat.py \
        --is_training 1 \
        --root_path ./data/ \
        --data_path Sanyo.csv \
        --model_id Sanyo_96'_'$pred_len \
        --model $model_name \
        --data custom \
        --features MS \
        --seq_len 96 \
        --pred_len $pred_len \
        --des 'Exp' \
        --itr 1 >logs/LongForecasting/$model_name'_Sanyo_'$pred_len.log

      python -u run_stat.py \
        --is_training 1 \
        --root_path ./data/ \
        --data_path Solibro.csv \
        --model_id Solibro_96'_'$pred_len \
        --model $model_name \
        --data custom \
        --features MS \
        --seq_len 96 \
        --pred_len $pred_len \
        --des 'Exp' \
        --itr 1 >logs/LongForecasting/$model_name'_Solibro_'$pred_len.log
  done
done