# Neur3D Dataset
CUDA_VISIBLE_DEVICES=0 python train.py -s /root/datasets/Neu3D/coffee_martini -m output_hash/coffee_martini --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth
CUDA_VISIBLE_DEVICES=0 python train.py -s /root/datasets/Neu3D/cook_spinach -m output_hash/cook_spinach --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth
CUDA_VISIBLE_DEVICES=0 python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_hash/cut_roasted_beef --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth
CUDA_VISIBLE_DEVICES=0 python train.py -s /root/datasets/Neu3D/flame_steak -m output_hash/flame_steak --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth
CUDA_VISIBLE_DEVICES=0 python train.py -s /root/datasets/Neu3D/sear_steak -m output_hash/sear_steak --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth
CUDA_VISIBLE_DEVICES=0 python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_hash/flame_salmon_1 --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_velocity_smooth

CUDA_VISIBLE_DEVICES=0 python render.py -m output_hash/coffee_martini --mode render
CUDA_VISIBLE_DEVICES=0 python render.py -m output_hash/cook_spinach --mode render
CUDA_VISIBLE_DEVICES=0 python render.py -m output_hash/cut_roasted_beef --mode render
CUDA_VISIBLE_DEVICES=0 python render.py -m output_hash/flame_steak --mode render
CUDA_VISIBLE_DEVICES=0 python render.py -m output_hash/sear_steak --mode render
CUDA_VISIBLE_DEVICES=0 python render.py -m output_hash/flame_salmon_1 --mode render

CUDA_VISIBLE_DEVICES=0 python metrics.py -m output_hash/coffee_martini
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output_hash/cook_spinach
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output_hash/cut_roasted_beef
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output_hash/flame_steak
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output_hash/sear_steak
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output_hash/flame_salmon_1

