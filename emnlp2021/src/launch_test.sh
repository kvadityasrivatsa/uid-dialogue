python3 pbft_scores.py \
    --data_path ../../conversation_corpora/mini_sent_empathetic_dialogues_sent_tokenized.csv \
    --out_path ./launch_test_output \
    --model_name "roberta" \
    --right_context 4 \
    --max_seq_len 64 \
    --add_speaker_ids \
    --per_gpu_batch_size 1 \
    --seed 7777
