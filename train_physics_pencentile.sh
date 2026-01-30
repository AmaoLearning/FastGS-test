# Neur3D Dataset
python train.py -s /root/datasets/Neu3D/coffee_martini -m output_phy/coffee_martini_50 --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_div_mask --div_percentile 50 --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/cook_spinach -m output_phy/cook_spinach_50 --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_div_mask --div_percentile 50 --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_phy/cut_roasted_beef_50 --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_div_mask --div_percentile 50 --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/flame_steak -m output_phy/flame_steak_50 --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_div_mask --div_percentile 50 --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/sear_steak -m output_phy/sear_steak_50 --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_div_mask --div_percentile 50 --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0
python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_phy/flame_salmon_1_50 --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_div_mask --div_percentile 50 --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0

python render.py -m output_phy/coffee_martini_50 --mode render
python render.py -m output_phy/cook_spinach_50 --mode render
python render.py -m output_phy/cut_roasted_beef_50 --mode render
python render.py -m output_phy/flame_steak_50 --mode render
python render.py -m output_phy/sear_steak_50 --mode render
python render.py -m output_phy/flame_salmon_1_50 --mode render

python metrics.py -m output_phy/coffee_martini_50
python metrics.py -m output_phy/cook_spinach_50
python metrics.py -m output_phy/cut_roasted_beef_50
python metrics.py -m output_phy/flame_steak_50
python metrics.py -m output_phy/sear_steak_50
python metrics.py -m output_phy/flame_salmon_1_50

