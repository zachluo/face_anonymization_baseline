set -ex
python train.py --dataroot /p300/dataset/casia \
                --name face_anonymize \
                --model pix2pix \
                --netG resnet_9blocks \
                --lambda_L1 0 \
                --dataset_mode casia \
                --landmark_path casia_landmark.txt \
                --sphere_model_path ./pretrain_model/sphere20a_20171020.pth \
                --gpu_ids 0,1,2,3 \
                --display_port 31190 \
                --display_server http://10.10.10.100 \
                --num_threads 32 \
                --batch_size 32
