set -ex
python train.py --dataroot /p300/project/download-celebA-HQ/celebA-HQ-256 \
                --model condition_pix2pix_nolandmark \
                --netG local \
                --netD n_layers \
                --n_layers_D 6 \
                --dataset_mode celeba_hq \
                --celeba_id_list /p300/project/download-celebA-HQ/celebA/Anno/identity_CelebA.txt \
                --celeba_hq_list /p300/project/download-celebA-HQ/image_list.txt \
                --n_classes_C 6217 \
                --sphere_model_path ./pretrain_model/sphere20a_20171020.pth \
                --gpu_ids 0,1,2,3 \
                --display_port 31190 \
                --display_server http://10.10.10.100 \
                --num_threads 32 \
                --batch_size 128 \
                --lambda_rec 10\
                --lambda_id 0 \
                --lambda_fr 5 \
                --lambda_GAN 1 \
                --margin -1 \
                --n_downsampling_global 2 \
                --n_blocks_global 6 \
                --n_local_enhancers 1 \
                --n_blocks_local 3 \
                --fill_percent 1 \
                --name celeba_hq_rec_10_id_0_fr_5_GAN_1_margin_-1_scale_128_wgan_bn_local2 \
                --display_env celeba_hq_rec_10_id_0_fr_5_GAN_1_margin_-1_scale_128_wgan_bn_local2\
                --load_size 128 \
                --norm batch \
                --gan_mode wgangp \
                --epoch_fix_global 100 \
                --load_pretrain celeba_hq_rec_10_id_0_fr_5_GAN_1_margin_-1_scale_64_wgan_bn_global2 \
                --continue_train




