# NeRF-DS Dataset
python train.py -s /root/datasets/nerf-ds/as_novel_view -m output_dynamic_mask/as_novel_view --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/basin_novel_view -m output_dynamic_mask/basin_novel_view --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/bell_novel_view -m output_dynamic_mask/bell_novel_view --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/cup_novel_view -m output_dynamic_mask/cup_novel_view --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/plate_novel_view -m output_dynamic_mask/plate_novel_view --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/press_novel_view -m output_dynamic_mask/press_novel_view --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/sieve_novel_view -m output_dynamic_mask/sieve_novel_view --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp

python render.py -m output_dynamic_mask/as_novel_view --mode render
python render.py -m output_dynamic_mask/basin_novel_view --mode render
python render.py -m output_dynamic_mask/bell_novel_view --mode render
python render.py -m output_dynamic_mask/cup_novel_view --mode render
python render.py -m output_dynamic_mask/plate_novel_view --mode render
python render.py -m output_dynamic_mask/press_novel_view --mode render
python render.py -m output_dynamic_mask/sieve_novel_view --mode render

python metrics.py -m output_dynamic_mask/as_novel_view
python metrics.py -m output_dynamic_mask/basin_novel_view
python metrics.py -m output_dynamic_mask/bell_novel_view
python metrics.py -m output_dynamic_mask/cup_novel_view
python metrics.py -m output_dynamic_mask/plate_novel_view
python metrics.py -m output_dynamic_mask/press_novel_view
python metrics.py -m output_dynamic_mask/sieve_novel_view

