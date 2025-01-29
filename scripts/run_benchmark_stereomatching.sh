#!/bin/bash
# run script : bash run_benchmark_stereomatching.sh

# PSMNet
CKPT_RGB="./checkpoints/MS2_SM_PSMNet_RGB_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_SM_PSMNet_THR_ckpt.ckpt"

SAVE_RGB="./result/PSMNet/rgb"
SAVE_THR="./result/PSMNet/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_THR}

CONFIG="configs/StereoSupDepth/PSMNet.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
	echo "Seq_name : ${SEQ}"
	CUDA_VISIBLE_DEVICES=0 python test_disparity.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
	CUDA_VISIBLE_DEVICES=0 python test_disparity.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# GWCNet
CKPT_RGB="./checkpoints/MS2_SM_GWCNet_RGB_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_SM_GWCNet_THR_ckpt.ckpt"

SAVE_RGB="./result/GWCNet/rgb"
SAVE_THR="./result/GWCNet/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_THR}

CONFIG="configs/StereoSupDepth/GWCNet.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
	echo "Seq_name : ${SEQ}"
	CUDA_VISIBLE_DEVICES=0 python test_disparity.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
	CUDA_VISIBLE_DEVICES=0 python test_disparity.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# AANet
CKPT_RGB="./checkpoints/MS2_SM_AANet_RGB_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_SM_AANet_THR_ckpt.ckpt"

SAVE_RGB="./result/AANet/rgb"
SAVE_THR="./result/AANet/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_THR}

CONFIG="configs/StereoSupDepth/AANet.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
	echo "Seq_name : ${SEQ}"
	CUDA_VISIBLE_DEVICES=0 python test_disparity.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
	CUDA_VISIBLE_DEVICES=0 python test_disparity.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

# ACVNet
CKPT_RGB="./checkpoints/MS2_SM_ACVNet_RGB_ckpt.ckpt"
CKPT_THR="./checkpoints/MS2_SM_ACVNet_THR_ckpt.ckpt"

SAVE_RGB="./result/ACVNet/rgb"
SAVE_THR="./result/ACVNet/thr"

mkdir -p ${SAVE_RGB}
mkdir -p ${SAVE_THR}

CONFIG="configs/StereoSupDepth/ACVNet.yaml"

SEQS=("test_day" "test_night" "test_rain")
for SEQ in ${SEQS[@]}; do
	echo "Seq_name : ${SEQ}"
	CUDA_VISIBLE_DEVICES=0 python test_disparity.py --config ${CONFIG} --ckpt_path ${CKPT_RGB} --test_env ${SEQ} --modality rgb --save_dir ${SAVE_RGB}/${SEQ} >> ${SAVE_RGB}_result.txt
	CUDA_VISIBLE_DEVICES=0 python test_disparity.py --config ${CONFIG} --ckpt_path ${CKPT_THR} --test_env ${SEQ} --modality thr --save_dir ${SAVE_THR}/${SEQ} >> ${SAVE_THR}_result.txt
done

