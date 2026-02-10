# Neur3D Dataset
#python train.py -s /root/datasets/hypernerf/interp_espresso -m output_dynamic_mask/interp_espresso --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
#python train.py -s /root/datasets/hypernerf/interp_americano -m output_dynamic_mask/interp_americano --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
#python train.py -s /root/datasets/hypernerf/interp_split-cookie -m output_dynamic_mask/interp_split-cookie --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
#python train.py -s /root/datasets/hypernerf/vrig-chicken -m output_dynamic_mask/vrig-chicken --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
#python train.py -s /root/datasets/hypernerf/interp_torchocolate -m output_dynamic_mask/interp_torchocolate --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
#python train.py -s /root/datasets/hypernerf/interp_cut-lemon1 -m output_dynamic_mask/interp_cut-lemon1 --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
#python train.py -s /root/datasets/hypernerf/interp_hand1-dense-v2 -m output_dynamic_mask/interp_hand1-dense-v2 --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
#python train.py -s /root/datasets/hypernerf/vrig-3dprinter -m output_dynamic_mask/vrig-3dprinter --eval --iterations 30000 --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
#
#python render.py -m output_dynamic_mask/interp_espresso --mode render
#python render.py -m output_dynamic_mask/interp_americano --mode render
#python render.py -m output_dynamic_mask/interp_split-cookie --mode render
#python render.py -m output_dynamic_mask/vrig-chicken --mode render
#python render.py -m output_dynamic_mask/interp_torchocolate --mode render
python render.py -m output_dynamic_mask/interp_cut-lemon1 --mode render
#python render.py -m output_dynamic_mask/interp_hand1-dense-v2 --mode render
#python render.py -m output_dynamic_mask/vrig-3dprinter --mode render
#
#python metrics.py -m output_dynamic_mask/interp_espresso
#python metrics.py -m output_dynamic_mask/interp_americano
#python metrics.py -m output_dynamic_mask/interp_split-cookie
#python metrics.py -m output_dynamic_mask/vrig-chicken
#python metrics.py -m output_dynamic_mask/interp_torchocolate
python metrics.py -m output_dynamic_mask/interp_cut-lemon1
#python metrics.py -m output_dynamic_mask/interp_hand1-dense-v2
#python metrics.py -m output_dynamic_mask/vrig-3dprinter

