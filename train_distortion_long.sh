CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck python train.py -s /root/datasets/tandt_db/tandt/truck -m output/truck_dist_dentil_10000  --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --densify_until_iter 10000 --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --lambda_dist 100 --use_distortion_loss True
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck python train.py -s /root/datasets/tandt_db/tandt/truck -m output/truck_dist_dentil_20000  --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --densify_until_iter 20000 --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --lambda_dist 100 --use_distortion_loss True
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck python train.py -s /root/datasets/tandt_db/tandt/truck -m output/truck_dist_dentil_25000  --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --densify_until_iter 25000 --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --lambda_dist 100 --use_distortion_loss True
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck python train.py -s /root/datasets/tandt_db/tandt/truck -m output/truck_dentil_10000  --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --densify_until_iter 10000 --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --lambda_dist 100
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck python train.py -s /root/datasets/tandt_db/tandt/truck -m output/truck_dentil_20000  --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --densify_until_iter 20000 --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --lambda_dist 100
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck python train.py -s /root/datasets/tandt_db/tandt/truck -m output/truck_dentil_25000  --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --densify_until_iter 25000 --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --lambda_dist 100
#CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=train python train.py -s /root/datasets/tandt_db/tandt/train -m output/train_dist_3dgs --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7 --lambda_dist 100 --use_distortion_loss True
#CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=playroom python train.py -s /root/datasets/tandt_db/db/playroom -m output/playroom_dist_3dgs --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --densify_until_iter 20000 --highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --lambda_dist 1000 --use_distortion_loss True
#CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=drjohnson python train.py -s /root/datasets/tandt_db/db/drjohnson -m output/drjohnson_dist_3dgs --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0025 --grad_abs_thresh 0.0012 --dense 0.013 --mult 0.7 --lambda_dist 1000 --use_distortion_loss True

CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_dist_dentil_10000 --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_dist_dentil_20000 --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_dist_dentil_25000 --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_dentil_10000 --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_dentil_20000 --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_dentil_25000 --skip_train --mult 0.7
#CUDA_VISIBLE_DEVICES=0 python render.py -m output/train_dist_3dgs --skip_train --mult 0.7
#CUDA_VISIBLE_DEVICES=0 python render.py -m output/playroom_dist_3dgs --skip_train --mult 0.7
#CUDA_VISIBLE_DEVICES=0 python render.py -m output/drjohnson_dist_3dgs --skip_train --mult 0.7

CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_dist_dentil_10000
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_dist_dentil_20000
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_dist_dentil_25000
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_dentil_10000
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_dentil_20000
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_dentil_25000
#CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/train_dist_3dgs
#CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/playroom_dist_3dgs
#CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/drjohnson_dist_3dgs