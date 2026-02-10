# Neur3D Dataset
python train.py -s /root/datasets/Neu3D/coffee_martini -m output_dynamic_mask/coffee_martini_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp --dense 0.1
python train.py -s /root/datasets/Neu3D/cook_spinach -m output_dynamic_mask/cook_spinach_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp --dense 0.1
python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_dynamic_mask/cut_roasted_beef_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp --dense 0.1
python train.py -s /root/datasets/Neu3D/flame_steak -m output_dynamic_mask/flame_steak_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp --dense 0.1
python train.py -s /root/datasets/Neu3D/sear_steak -m output_dynamic_mask/sear_steak_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp --dense 0.1
python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_dynamic_mask/flame_salmon_1_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp --dense 0.1

python render.py -m output_dynamic_mask/coffee_martini_larger_dense --mode render
python render.py -m output_dynamic_mask/cook_spinach_larger_dense --mode render
python render.py -m output_dynamic_mask/cut_roasted_beef_larger_dense --mode render
python render.py -m output_dynamic_mask/flame_steak_larger_dense --mode render
python render.py -m output_dynamic_mask/sear_steak_larger_dense --mode render
python render.py -m output_dynamic_mask/flame_salmon_1_larger_dense --mode render

python metrics.py -m output_dynamic_mask/coffee_martini_larger_dense
python metrics.py -m output_dynamic_mask/cook_spinach_larger_dense
python metrics.py -m output_dynamic_mask/cut_roasted_beef_larger_dense
python metrics.py -m output_dynamic_mask/flame_steak_larger_dense
python metrics.py -m output_dynamic_mask/sear_steak_larger_dense
python metrics.py -m output_dynamic_mask/flame_salmon_1_larger_dense

