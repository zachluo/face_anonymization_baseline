set -ex
CUDA_VISIBLE_DEVICES=1 python test.py \
                --dataroot /p300/dataset/casia \
                --model pix2pix \
                --netG resnet_9blocks \
                --netD multiD \
                --dataset_mode casia \
                --landmark_path casia_landmark.txt \
                --gpu_ids 0 \
                --num_threads 32 \
                --batch_size 24 \
                --name rec_10_id_0_fr_0_GAN_1 \
                --load_size 128 \
                --normG batch \
                --normD none \
                --load_pretrain rec_10_id_0_fr_0_GAN_1 \
                --epoch 5 \
                --eval \
                --num_test 1 \
