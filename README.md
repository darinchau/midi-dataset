Training

```
accelerate launch -m scripts.train_cp_tokenizer --n_encoder_blocks 8 --n_decoder_blocks 8 --hidden_dims 256  --max_seq_length 6144 --num_embeddings 4096 --use_checkpoint --batch_size 2 --grad_accumulation_steps 32 --num_workers 8 --limit_gpu_temp './resources/gpu.json'
```
