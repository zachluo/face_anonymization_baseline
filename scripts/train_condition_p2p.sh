set -ex
python train.py --dataroot /p300/dataset/casia \
                --model condition_pix2pix \
                --netG resnet_6blocks \
                --dataset_mode casia \
                --landmark_path casia_landmark.txt \
                --sphere_model_path ./pretrain_model/sphere20a_20171020.pth \
                --gpu_ids 0,1,2,3 \
                --display_port 31190 \
                --display_server http://10.10.10.100 \
                --num_threads 32 \
                --batch_size 128 \
                --lambda_rec 10 \
                --lambda_id 0 \
                --lambda_fr 1 \
                --lambda_GAN 0 \
                --lambda_condition 0 \
                --name rec_10_id_0_fr_1_GAN_0_con_0_scale_128_bn \
                --display_env rec_10_id_0_fr_1_GAN_0_con_0_scale_128_bn \
                --load_size 128 \
                --norm batch \



