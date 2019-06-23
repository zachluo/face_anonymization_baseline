set -ex
python test.py --dataroot /p300/project/download-celebA-HQ/celebA-HQ-256 \
                --model condition_pix2pix_nolandmark \
                --netG global \
                --netD n_layers \
                --n_layers_D 6 \
                --dataset_mode celeba_hq \
                --celeba_id_list /p300/project/download-celebA-HQ/celebA/Anno/identity_CelebA.txt \
                --celeba_hq_list /p300/project/download-celebA-HQ/image_list.txt \
                --gpu_ids 0 \
                --batch_size 64 \
                --name celeba_hq_rec_10_id_0_fr_5_GAN_1_margin_-1_scale_64_wgan_bn_global2 \
                --load_pretrain celeba_hq_rec_10_id_0_fr_5_GAN_1_margin_-1_scale_64_wgan_bn_global2 \
                --load_size 64 \
                --n_downsampling_global 2 \
                --n_blocks_global 6 \
                --n_local_enhancers 1 \
                --n_blocks_local 3 \
                --fill_percent 1 \
                --display_winsize 64 \
                --norm batch \
                --num_test 5000



