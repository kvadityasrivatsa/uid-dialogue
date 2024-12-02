python3 pbft_t5_scores.py \
    --data_path emowoz.csv \
    --out_path ./launch_test_output/emowoz_t5 \
    --model_name t5 \
    --model_path ../t5-emowoz_pairs-finetuned_final \
    --right_context 4 \
    --max_seq_len 64 \
    --add_speaker_ids \
    --per_gpu_batch_size 1 \
    --seed 7777