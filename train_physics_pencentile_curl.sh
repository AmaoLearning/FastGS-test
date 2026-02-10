# Neur3D Dataset
python train.py -s /root/datasets/Neu3D/coffee_martini -m output_phy/coffee_martini_50_curl_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0 --dense 0.1
python train.py -s /root/datasets/Neu3D/cook_spinach -m output_phy/cook_spinach_50_curl_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0 --dense 0.1
python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_phy/cut_roasted_beef_50_curl_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0 --dense 0.1
python train.py -s /root/datasets/Neu3D/flame_steak -m output_phy/flame_steak_50_curl_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0 --dense 0.1
python train.py -s /root/datasets/Neu3D/sear_steak -m output_phy/sear_steak_50_curl_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0 --dense 0.1
python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_phy/flame_salmon_1_50_curl_larger_dense --eval --iterations 30000 --use_velocity --use_dynamic_mask --velocity_network_type hash --use_physics_densify --use_curl_mask --curl_percentile 50 --physics_clone_eta 0.2 --physics_split_scale_factor 2.0 --dense 0.1

python render.py -m output_phy/coffee_martini_50_curl_larger_dense --mode render
python render.py -m output_phy/cook_spinach_50_curl_larger_dense --mode render
python render.py -m output_phy/cut_roasted_beef_50_curl_larger_dense --mode render
python render.py -m output_phy/flame_steak_50_curl_larger_dense --mode render
python render.py -m output_phy/sear_steak_50_curl_larger_dense --mode render
python render.py -m output_phy/flame_salmon_1_50_curl_larger_dense --mode render

python metrics.py -m output_phy/coffee_martini_50_curl_larger_dense
python metrics.py -m output_phy/cook_spinach_50_curl_larger_dense
python metrics.py -m output_phy/cut_roasted_beef_50_curl_larger_dense
python metrics.py -m output_phy/flame_steak_50_curl_larger_dense
python metrics.py -m output_phy/sear_steak_50_curl_larger_dense
python metrics.py -m output_phy/flame_salmon_1_50_curl_larger_dense

