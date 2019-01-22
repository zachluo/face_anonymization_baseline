set -ex
python train.py --dataroot /p300/dataset/casia \
                --model pix2pix \
                --netG resnet_9blocks \
                --dataset_mode casia \
                --landmark_path casia_landmark.txt \
                --sphere_model_path ./pretrain_model/sphere20a_20171020.pth \
                --gpu_ids 0,1 \
                --display_port 31190 \
                --display_server http://10.10.10.100 \
                --num_threads 32 \
                --batch_size 16 \
                --lambda_rec 0\
                --lambda_id 0 \
                --lambda_fr 0 \
                --lambda_GAN 1 \
                --name rec_0_id_0_fr_0_WGAN_1 \
                --display_env rec_0_id_0_fr_0_WGAN_1 \
                --gan_mode wgangp
