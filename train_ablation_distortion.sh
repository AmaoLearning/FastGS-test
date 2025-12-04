CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck python train.py -s /root/datasets/tandt_db/tandt/truck -m output/truck_dist_vcd --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --prune_as_fastgs False
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck python train.py -s /root/datasets/tandt_db/tandt/truck -m output/truck_dist_vcp --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 --densify_as_fastgs False
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=train python train.py -s /root/datasets/tandt_db/tandt/train -m output/train_dist_vcd --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7 --prune_as_fastgs False
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=train python train.py -s /root/datasets/tandt_db/tandt/train -m output/train_dist_vcp --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7 --densify_as_fastgs False
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=playroom python train.py -s /root/datasets/tandt_db/db/playroom -m output/playroom_dist_vcd --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --prune_as_fastgs False
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=playroom python train.py -s /root/datasets/tandt_db/db/playroom -m output/playroom_dist_vcp --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --densify_as_fastgs False
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=drjohnson python train.py -s /root/datasets/tandt_db/db/drjohnson -m output/drjohnson_dist_vcd --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0025 --grad_abs_thresh 0.0012 --dense 0.013 --mult 0.7 --prune_as_fastgs False
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=drjohnson python train.py -s /root/datasets/tandt_db/db/drjohnson -m output/drjohnson_dist_vcp --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0025 --grad_abs_thresh 0.0012 --dense 0.013 --mult 0.7 --densify_as_fastgs False

CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_dist_vcd --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/truck_dist_vcp --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/train_dist_vcd --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/train_dist_vcp --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/playroom_dist_vcd --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/playroom_dist_vcp --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/drjohnson_dist_vcd --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 python render.py -m output/drjohnson_dist_vcp --skip_train --mult 0.7

CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_dist_vcd
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/truck_dist_vcp
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/train_dist_vcd
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/train_dist_vcp
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/playroom_dist_vcd
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/playroom_dist_vcp
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/drjohnson_dist_vcd
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/drjohnson_dist_vcp