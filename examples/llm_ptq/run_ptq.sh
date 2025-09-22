# cant export int8_sq and int4_awq to HF

export HF_HOME=/home/scratch.shreyasm_gpu/hf_cache

# Baseline int8_sq export

# python hf_ptq.py --pyt_ckpt_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b \
#     --verbose \
#     --export_fmt tensorrt_llm \
#     --kv_cache_qformat none \
#     --dataset magpie \
#     --batch_size 4 \
#     --qformat nvfp4 \
#     --export_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b_nvfp4_magpie_bf16kv

# custom

# python hf_ptq.py --pyt_ckpt_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b \
#     --verbose \
#     --export_fmt tensorrt_llm \
#     --kv_cache_qformat fp8 \
#     --dataset magpie \
#     --batch_size 4 \
#     --qformat nvfp4_custom_3 \
#     --export_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b_nvfp4_custom_3_magpie


# Auto quantize

# python hf_ptq.py --pyt_ckpt_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b \
#     --verbose \
#     --export_fmt tensorrt_llm \
#     --kv_cache_qformat none \
#     --auto_quantize_bits 9.0 \
#     --dataset cnn_dailymail \
#     --batch_size 4 \
#     --qformat int8_sq,none \
#     --export_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b_int8_sq_auto_quantized_cnn_dailymail

CUDA_VISIBLE_DEVICES=0,1,2,3 python hf_ptq.py --pyt_ckpt_path /home/scratch.shreyasm_gpu/models/Llama-3_3-Nemotron-Super-49B-v1_5 \
     --verbose \
     --export_fmt tensorrt_llm \
     --kv_cache_qformat none \
     --auto_quantize_bits 9.0 \
     --dataset magpie \
     --batch_size 4 \
     --qformat int8_sq,none \
     --trust_remote_code \
     --export_path /home/scratch.shreyasm_gpu/models/llama-3.3-super-49b-v1_5_int8_sq_auto_quantized_magpie

#CUDA_VISIBLE_DEVICES=0,1,2,3 python hf_ptq.py --pyt_ckpt_path /home/scratch.shreyasm_gpu/models/llama-3.1-foxbrain-70b_vhf-d60d786-tool-calling \
#    --verbose \
#    --export_fmt tensorrt_llm \
#    --kv_cache_qformat fp8 \
#    --dataset magpie \
#    --batch_size 4 \
#    --qformat fp8 \
#    --trust_remote_code \
#    --export_path /home/scratch.shreyasm_gpu/models/llama-3.1-foxbrain-70b_vhf-d60d786-tool-calling_fp8_magpie
