# NeRF-DS Dataset
python train.py -s /root/datasets/nerf-ds/vrig_as -m output_dynamic_mask/vrig_as --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/vrig_basin -m output_dynamic_mask/vrig_basin --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/vrig_bell -m output_dynamic_mask/vrig_bell --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/vrig_cup -m output_dynamic_mask/vrig_cup --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/vrig_plate -m output_dynamic_mask/vrig_plate --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/vrig_press -m output_dynamic_mask/vrig_press --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/nerf-ds/vrig_sieve -m output_dynamic_mask/vrig_sieve --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp

python render.py -m output_dynamic_mask/vrig_as --mode render
python render.py -m output_dynamic_mask/vrig_basin --mode render
python render.py -m output_dynamic_mask/vrig_bell --mode render
python render.py -m output_dynamic_mask/vrig_cup --mode render
python render.py -m output_dynamic_mask/vrig_plate --mode render
python render.py -m output_dynamic_mask/vrig_press --mode render
python render.py -m output_dynamic_mask/vrig_sieve --mode render

python metrics.py -m output_dynamic_mask/vrig_as
python metrics.py -m output_dynamic_mask/vrig_basin
python metrics.py -m output_dynamic_mask/vrig_bell
python metrics.py -m output_dynamic_mask/vrig_cup
python metrics.py -m output_dynamic_mask/vrig_plate
python metrics.py -m output_dynamic_mask/vrig_press
python metrics.py -m output_dynamic_mask/vrig_sieve

