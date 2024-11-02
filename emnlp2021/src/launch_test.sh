python pbft_scores.py \
    --data_path ../data/true/pb_gpt2-ft.csv \
    --out_path ./launch_test_output \
    --model_name "roberta" \
    --model_path "<path>" \
    --right_context 4 \
    --max_seq_len 64 \
    --add_speaker_ids \
    --per_gpu_batch_size 1 \
    --seed 7777 
