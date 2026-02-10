# Neur3D Dataset
#python train.py -s /root/datasets/Neu3D/coffee_martini -m output_hash/coffee_martini_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth --dense 0.1
#python train.py -s /root/datasets/Neu3D/cook_spinach -m output_hash/cook_spinach_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth --dense 0.1
#python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_hash/cut_roasted_beef_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth --dense 0.1
#python train.py -s /root/datasets/Neu3D/flame_steak -m output_hash/flame_steak_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth --dense 0.1
#python train.py -s /root/datasets/Neu3D/sear_steak -m output_hash/sear_steak_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth --dense 0.1
#python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_hash/flame_salmon_1_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth --dense 0.1
#
#python render.py -m output_hash/coffee_martini_larger_dense --mode render
#python render.py -m output_hash/cook_spinach_larger_dense --mode render
#python render.py -m output_hash/cut_roasted_beef_larger_dense --mode render
#python render.py -m output_hash/flame_steak_larger_dense --mode render
#python render.py -m output_hash/sear_steak_larger_dense --mode render
#python render.py -m output_hash/flame_salmon_1_larger_dense --mode render

python metrics.py -m output_hash/coffee_martini_larger_dense
python metrics.py -m output_hash/cook_spinach_larger_dense
python metrics.py -m output_hash/cut_roasted_beef_larger_dense
python metrics.py -m output_hash/flame_steak_larger_dense
python metrics.py -m output_hash/sear_steak_larger_dense
python metrics.py -m output_hash/flame_salmon_1_larger_dense

#python train.py -s /root/datasets/Neu3D/coffee_martini -m output_hash/coffee_martini_without_smooth_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --dense 0.1
#python train.py -s /root/datasets/Neu3D/cook_spinach -m output_hash/cook_spinach_without_smooth_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --dense 0.1
#python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_hash/cut_roasted_beef_without_smooth_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --dense 0.1
#python train.py -s /root/datasets/Neu3D/flame_steak -m output_hash/flame_steak_without_smooth_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --dense 0.1
#python train.py -s /root/datasets/Neu3D/sear_steak -m output_hash/sear_steak_without_smooth_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --dense 0.1
#python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_hash/flame_salmon_1_without_smooth_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --dense 0.1
#
#python render.py -m output_hash/coffee_martini_without_smooth_larger_dense --mode render
#python render.py -m output_hash/cook_spinach_without_smooth_larger_dense --mode render
#python render.py -m output_hash/cut_roasted_beef_without_smooth_larger_dense --mode render
#python render.py -m output_hash/flame_steak_without_smooth_larger_dense --mode render
#python render.py -m output_hash/sear_steak_without_smooth_larger_dense --mode render
#python render.py -m output_hash/flame_salmon_1_without_smooth_larger_dense --mode render

python metrics.py -m output_hash/coffee_martini_without_smooth_larger_dense
python metrics.py -m output_hash/cook_spinach_without_smooth_larger_dense
python metrics.py -m output_hash/cut_roasted_beef_without_smooth_larger_dense
python metrics.py -m output_hash/flame_steak_without_smooth_larger_dense
python metrics.py -m output_hash/sear_steak_without_smooth_larger_dense
python metrics.py -m output_hash/flame_salmon_1_without_smooth_larger_dense
