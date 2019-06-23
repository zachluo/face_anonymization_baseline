set -ex
python train.py --dataroot /p300/dataset/casia \
                --model baseline \
                --netG resnet_9blocks \
                --netD multiD \
                --dataset_mode casia_baseline \
                --landmark_path casia_landmark.txt \
                --gpu_ids 3 \
                --display_port 31190 \
                --display_server http://10.10.10.100 \
                --num_threads 32 \
                --batch_size 48 \
                --lambda_rec 10 \
                --lambda_GAN 1 \
                --name baseline_edge_wgan_MD \
                --display_env baseline_edge_wgan_MD \
                --load_size 128 \
                --baseline edge \
                --gan_mode wgangp \
                --normD none \
                --normG batch



