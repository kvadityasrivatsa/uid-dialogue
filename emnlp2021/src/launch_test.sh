python3 pbft_scores.py \
    --data_path ../../conversation_corpora/empathetic_dialogues_sent_tokenized.csv \
    --out_path ./launch_test_output/empdial \
    --model_name roberta \
    --model_path ../roberta-empathetic_dialogues_sent_tokenized-finetuned_final \
    --right_context 4 \
    --max_seq_len 64 \
    --add_speaker_ids \
    --per_gpu_batch_size 1 \
    --seed 7777