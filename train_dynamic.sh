# Neur3D Dataset
python train.py -s /root/datasets/Neu3D/coffee_martini -m output_dynamic_mask/coffee_martini --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/Neu3D/cook_spinach -m output_dynamic_mask/cook_spinach --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_dynamic_mask/cut_roasted_beef --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/Neu3D/flame_steak -m output_dynamic_mask/flame_steak --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/Neu3D/sear_steak -m output_dynamic_mask/sear_steak --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_dynamic_mask/flame_salmon_1 --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp

python render.py -m output_dynamic_mask/coffee_martini --mode render
python render.py -m output_dynamic_mask/cook_spinach --mode render
python render.py -m output_dynamic_mask/cut_roasted_beef --mode render
python render.py -m output_dynamic_mask/flame_steak --mode render
python render.py -m output_dynamic_mask/sear_steak --mode render
python render.py -m output_dynamic_mask/flame_salmon_1 --mode render

python metrics.py -m output_dynamic_mask/coffee_martini
python metrics.py -m output_dynamic_mask/cook_spinach
python metrics.py -m output_dynamic_mask/cut_roasted_beef
python metrics.py -m output_dynamic_mask/flame_steak
python metrics.py -m output_dynamic_mask/sear_steak
python metrics.py -m output_dynamic_mask/flame_salmon_1

