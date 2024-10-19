python ft_pb_sentence_entropy.py \
    --data_path ../data/true/pb_gpt2-ft.csv \
    --out_path ./launch_test_output \
    --model_name "gpt2" \
    --model_path ./ft_gpt2/ \
    --right_context 4 \
    --max_seq_len 64 \
    --add_speaker_ids \
    --per_gpu_batch_size 64 \
    --learning_rate 5e-5 \
    --num_epochs 30 \
    --seed 42