set -ex
python train.py --dataroot /p300/dataset/casia \
                --model pix2pix_aligned \
                --netG resnet_6blocks \
                --dataset_mode casia \
                --landmark_path casia_landmark.txt \
                --sphere_model_path ./pretrain_model/sphere20a_20171020.pth \
                --gpu_ids 0,1,2,3 \
                --display_port 31190 \
                --display_server http://10.10.10.100 \
                --num_threads 32 \
                --batch_size 64 \
                --lambda_rec 0\
                --lambda_id 0 \
                --lambda_fr 1 \
                --lambda_GAN 1 \
                --fr_level 4 \
                --name xiuye_rec_0_id_0_fr_1_GAN_1_scale_256 \
                --display_env xiuye_rec_0_id_0_fr_1_GAN_1_scale_256 \
                --load_size 256 \
                --norm batch



