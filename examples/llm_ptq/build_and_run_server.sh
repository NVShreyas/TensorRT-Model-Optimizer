engine_dir=/tmp/trtllm_build/engine
pyt_dir=/home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b
quant_ckpt_dir=/home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b_int8_sq_auto_quantized_cnn_dailymail

tp_size=1
seq_len=131072

rm -r $engine_dir
mkdir $engine_dir
trtllm-build --checkpoint_dir  $quant_ckpt_dir --output_dir $engine_dir --max_batch_size 128 --max_seq_len ${seq_len} --max_num_tokens 8192 --gpus_per_node ${tp_size} --workers ${tp_size} --nccl_plugin auto --gpt_attention_plugin auto --gemm_plugin auto --context_fmha enable --kv_cache_type paged --remove_input_padding enable --tokens_per_block 64 --multiple_profiles enable --use_fused_mlp enable --use_paged_context_fmha enable

trtllm-serve ${engine_dir} \
    --tokenizer ${pyt_dir} \
    --host 0.0.0.0 \
    --port 8001 \
    --backend trt \
    --tp_size $tp_size \
    --pp_size 1 \
    --trust_remote_code \
    --kv_cache_free_gpu_memory_fraction 0.9 \
    --num_postprocess_workers 10 \
    --max_num_tokens 8192 \
    --max_batch_size 128


# workstation ip address - http://10.112.213.22:18001/v1/models
# ngc registry model upload-version "nvstaging/nim/llama-3.1-8b-instruct:"
# http://10.63.139.51:18001/v1/chat/completions 
# http://10.34.2.158:18001/v1/chat/completions - a100