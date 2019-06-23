set -ex
CUDA_VISIBLE_DEVICES=2 python eval.py --dataroot /p300/dataset/lfw \
                --model pix2pix \
                --netG resnet_9blocks \
                --dataset_mode lfw \
                --landmark_path lfw_landmark.txt \
                --gpu_ids 0 \
                --num_threads 32 \
                --batch_size 48 \
                --name rec_10_id_0_fr_0_GAN_1 \
                --load_pretrain rec_10_id_0_fr_0_GAN_1 \
                --load_size 128 \
                --normG batch \
                --display_winsize 128 \
                --serial_batches \
                --epoch 5 \
                --num_test 1000000 \
                --val_test test \
                --eval



