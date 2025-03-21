export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_DISABLE_ABORT
export NCCL_TIMEOUT=3600000
export CUDA_LAUNCH_BLOCKING=1
export NCCL_P2P_DISABLE=1

############### Finetune ################

export NUM_GPUS=8
export NNODES=1
export RANK=0
export MASTER_ADDR=0.0.0.0
export MASTER_PORT=8031

# Stage 2
PROMPT_VERSION="qwen_1_5"
RUN_NAME="ViLAMP-llava-qwen-FT"
PREV_STAGE_CHECKPOINT="models/ViLAMP-llava-qwen"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "NNODES: ${NNODES}"

ACCELERATE_CPU_AFFINITY=0 TOKENIZERS_PARALLELISM=false torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path "scripts/train/training_data.yaml" \
    --video_folder "dataset"\
    --mm_tunable_parts="mm_language_model,mm_mlp_adapter" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower "siglip-so400m-patch14-384" \
    --clip_model_path "clip-ViT-B-32/0_CLIPModel" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir "models/${RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --report_to none
    # --lora_enable True
exit 0;

# You can delete the sdpa attn_implementation if you want to use flash attn
