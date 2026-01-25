# Neur3D Dataset
python train.py -s /root/datasets/Neu3D/coffee_martini -m output_phy/coffee_martini --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --div_percentile 95 --curl_percentile 95 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/cook_spinach -m output_phy/cook_spinach --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --div_percentile 95 --curl_percentile 95 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_phy/cut_roasted_beef --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --div_percentile 95 --curl_percentile 95 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/flame_steak -m output_phy/flame_steak --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --div_percentile 95 --curl_percentile 95 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/sear_steak -m output_phy/sear_steak --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --div_percentile 95 --curl_percentile 95 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_phy/flame_salmon_1 --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --div_percentile 95 --curl_percentile 95 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0

python render.py -m output_phy/coffee_martini --mode render
python render.py -m output_phy/cook_spinach --mode render
python render.py -m output_phy/cut_roasted_beef --mode render
python render.py -m output_phy/flame_steak --mode render
python render.py -m output_phy/sear_steak --mode render
python render.py -m output_phy/flame_salmon_1 --mode render

python metrics.py -m output_phy/coffee_martini
python metrics.py -m output_phy/cook_spinach
python metrics.py -m output_phy/cut_roasted_beef
python metrics.py -m output_phy/flame_steak
python metrics.py -m output_phy/sear_steak
python metrics.py -m output_phy/flame_salmon_1

