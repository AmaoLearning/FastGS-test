# Neur3D Dataset
python train.py -s /remote-home/panyicheng/datasets/Neu3D/coffee_martini -m output_gaufre/coffee_martini --eval --iterations 30000 --lazy_load --deform_type 4dgs --log_deform_hist --num_dynamic_gaussians 100000 --cluster_w_xyz 1 --cluster_w_color 0.5 --cluster_w_motion 0.5
python train.py -s /remote-home/panyicheng/datasets/Neu3D/cook_spinach -m output_gaufre/cook_spinach --eval --iterations 30000 --lazy_load --deform_type 4dgs --log_deform_hist --num_dynamic_gaussians 100000 --cluster_w_xyz 1 --cluster_w_color 0.5 --cluster_w_motion 0.5
python train.py -s /remote-home/panyicheng/datasets/Neu3D/cut_roasted_beef -m output_gaufre/cut_roasted_beef --eval --iterations 30000 --lazy_load --deform_type 4dgs --log_deform_hist --num_dynamic_gaussians 100000 --cluster_w_xyz 1 --cluster_w_color 0.5 --cluster_w_motion 0.5
python train.py -s /remote-home/panyicheng/datasets/Neu3D/flame_steak -m output_gaufre/flame_steak --eval --iterations 30000 --lazy_load --deform_type 4dgs --log_deform_hist --num_dynamic_gaussians 100000 --cluster_w_xyz 1 --cluster_w_color 0.5 --cluster_w_motion 0.5
python train.py -s /remote-home/panyicheng/datasets/Neu3D/sear_steak -m output_gaufre/sear_steak --eval --iterations 30000 --lazy_load --deform_type 4dgs --log_deform_hist --num_dynamic_gaussians 100000 --cluster_w_xyz 1 --cluster_w_color 0.5 --cluster_w_motion 0.5
python train.py -s /remote-home/panyicheng/datasets/Neu3D/flame_salmon_1 -m output_gaufre/flame_salmon_1 --eval --iterations 30000 --lazy_load --deform_type 4dgs --log_deform_hist --num_dynamic_gaussians 100000 --cluster_w_xyz 1 --cluster_w_color 0.5 --cluster_w_motion 0.5

python render.py -m output_gaufre/coffee_martini --mode render --skip_train
python render.py -m output_gaufre/cook_spinach --mode render --skip_train
python render.py -m output_gaufre/cut_roasted_beef --mode render --skip_train
python render.py -m output_gaufre/flame_steak --mode render --skip_train
python render.py -m output_gaufre/sear_steak --mode render --skip_train
python render.py -m output_gaufre/flame_salmon_1 --mode render --skip_train

python metrics.py -m output_gaufre/coffee_martini
python metrics.py -m output_gaufre/cook_spinach
python metrics.py -m output_gaufre/cut_roasted_beef
python metrics.py -m output_gaufre/flame_steak
python metrics.py -m output_gaufre/sear_steak
python metrics.py -m output_gaufre/flame_salmon_1

