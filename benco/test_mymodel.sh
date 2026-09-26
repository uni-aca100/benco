base_dir="./eval_dir"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0 \
python ./test.py \
    --model MyModel \
    --MyModel_Customized_param 12345678 \
    --pre_trained_weights './static/mock_weights.pth' \
    --world_size 1 \
    --test_data_json "./test_datasets.json" \
    --checkpoint_path "./output_dir/" \
    --test_batch_size 1 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --model_sock "/tmp/socket/uvicorn.sock" \
    --model_req_path '/pred/' \
    --request_retry_timeout 300 \
    --request_retries 2 \
    --request_retry_initial_delay 20 \
    --request_retry_delay_factor 2 \
> >(tee "${base_dir}/logs.log") 2> >(tee "${base_dir}/error.log" >&2)