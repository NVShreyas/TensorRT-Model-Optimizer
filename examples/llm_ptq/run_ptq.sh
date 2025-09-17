# cant export int8_sq and int4_awq to HF

export HF_HOME=/home/hf_cache

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

python hf_ptq.py --pyt_ckpt_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b \
    --verbose \
    --export_fmt tensorrt_llm \
    --kv_cache_qformat none \
    --auto_quantize_bits 9.0 \
    --dataset cnn_dailymail \
    --batch_size 4 \
    --qformat int8_sq,none \
    --export_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b_int8_sq_auto_quantized_cnn_dailymail

# python hf_ptq.py --pyt_ckpt_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b \
#     --verbose \
#     --export_fmt tensorrt_llm \
#     --kv_cache_qformat none \
#     --auto_quantize_bits 8.0 \
#     --dataset magpie \
#     --batch_size 4 \
#     --qformat nvfp4 \
#     --export_path /home/shreyasm/workspace/models/llama-3.1-8b-instruct_vhf-8c22764-nim1.3b_nvfp4_auto_quantized_magpie