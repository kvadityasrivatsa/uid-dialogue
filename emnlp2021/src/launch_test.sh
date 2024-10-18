python pb_sentence_entropy.py \
    --data_path ../data/true/pb_gpt2-ft.csv \
    --out_path ./launch_test_output \
    --model_name "gpt2" \
    --right_context 4 \
    --max_seq_len 64 \
    --per_gpu_batch_size 1 \
    --seed 7777