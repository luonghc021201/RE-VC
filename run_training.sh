export CUDA_VISIBLE_DEVICES='0, 3'
LD_LIBRARY_PATH=/data/tantc/cuda-11.6/lib64:$LD_LIBRARY_PATH \
python_cmd=/data/miniconda3/envs/freevc/bin/python
case "$1" in
version1) # Train voice conversion with wavlm, on vctk dataset
    $python_cmd train_ms.py \
        -c configs/vits_vc_vctk.json \
        -m /data/luonghc/vits_vc_aug_wavlm_vctk_logs_try

;;
*)
    echo "Do not found mode, mode should be either 'original' or ..."
    esac