#!/bin/bash
# run script : bash run_benchmark_monodepth.sh

# DORN
CKPT_RGB="./checkpoints/MS2_MD_DORN_RGB_ckpt.ckpt"
CKPT_NIR="./checkpoints/MS2_MD_DORN_NIR_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_MD_DORN_THR_ckpt.ckpt"

SAVE_RGB="./result/DORN/rgb"
SAVE_NIR="./result/DORN/nir"
SAVE_THR="./result/DORN/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/DORN.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_NIR} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# BTS
CKPT_RGB="./checkpoints/MS2_MD_BTS_RGB_ckpt.ckpt"
CKPT_NIR="./checkpoints/MS2_MD_BTS_NIR_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_MD_BTS_THR_ckpt.ckpt"

SAVE_RGB="./result/BTS/rgb"
SAVE_NIR="./result/BTS/nir"
SAVE_THR="./result/BTS/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/BTS.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_NIR} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# AdaBins
CKPT_RGB="./checkpoints/MS2_MD_AdaBins_RGB_ckpt.ckpt"
CKPT_NIR="./checkpoints/MS2_MD_AdaBins_NIR_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_MD_AdaBins_THR_ckpt.ckpt"

SAVE_RGB="./result/AdaBins/rgb"
SAVE_NIR="./result/AdaBins/nir"
SAVE_THR="./result/AdaBins/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/AdaBins.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_NIR} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# DPT_large
CKPT_NIR="./checkpoints/MS2_MD_DPT_Large_NIR_ckpt.ckpt"
CKPT_RGB="./checkpoints/MS2_MD_DPT_Large_RGB_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_MD_DPT_Large_THR_ckpt.ckpt"

SAVE_RGB="./result/DPT_large/rgb"
SAVE_NIR="./result/DPT_large/nir"
SAVE_THR="./result/DPT_large/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/DPT_large.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_NIR} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# NewCRF
CKPT_NIR="./checkpoints/MS2_MD_NeWCRF_NIR_ckpt.ckpt"
CKPT_RGB="./checkpoints/MS2_MD_NeWCRF_RGB_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_MD_NeWCRF_THR_ckpt.ckpt"

SAVE_RGB="./result/NewCRF/rgb"
SAVE_NIR="./result/NewCRF/nir"
SAVE_THR="./result/NewCRF/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_NIR}
mkdir -p ${SAVE_THR}

CONFIG="configs/MonoSupDepth/NewCRF.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
    echo "Seq_name : ${SEQ}"
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_NIR} --test_env ${SEQ} --modality nir --save_dir ${SAVE_NIR}/${SEQ} >> ${SAVE_NIR}_result.txt
    CUDA_VISIBLE_DEVICES=0 python test_monodepth.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done
