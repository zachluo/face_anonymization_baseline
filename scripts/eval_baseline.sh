set -ex
CUDA_VISIBLE_DEVICES=2 python eval.py --dataroot /p300/dataset/lfw \
                --model baseline \
                --netG resnet_9blocks \
                --dataset_mode lfw_baseline \
                --landmark_path lfw_landmark.txt \
                --gpu_ids 0 \
                --num_threads 32 \
                --batch_size 48 \
                --name baseline_super_pixel_wgan_MD \
                --load_pretrain baseline_super_pixel_wgan_MD \
                --load_size 128 \
                --normG batch \
                --normD none \
                --baseline super_pixel \
                --display_winsize 128 \
                --serial_batches \
                --epoch 13 \
                --num_test 1000000 \
                --val_test test \
                --eval



