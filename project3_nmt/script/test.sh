
NGPU=1
CUDA_VISIBLE_DEVICES=$NGPU python inference.py \
  --root data \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 64 \
  --gradient_accumulation_step 1 \
  --checkpoint_dir checkpoint ;

