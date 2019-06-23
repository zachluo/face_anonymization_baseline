set -ex
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot /p300/dataset/casia \
                --model baseline \
                --netG resnet_9blocks \
                --dataset_mode casia_baseline \
                --landmark_path casia_landmark.txt \
                --gpu_ids 0 \
                --batch_size 48 \
                --name baseline_super_pixel_wgan_MD \
                --load_pretrain baseline_super_pixel_wgan_MD \
                --load_size 128 \
                --normG batch \
                --normD none \
                --baseline super_pixel \
                --display_winsize 128 \
                --serial_batches \
                --num_threads 64 \
                --epoch 13 \
                --num_test 1 \
                --eval \
                --val_test test



