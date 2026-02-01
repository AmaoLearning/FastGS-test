# D-NeRF Dataset
python train.py -s /root/datasets/dnerf/bouncingballs -m output_dynamic_mask/bouncingballs --eval --iterations 30000 --is_blender --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/dnerf/hellwarrior -m output_dynamic_mask/hellwarrior --eval --iterations 30000 --is_blender --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/dnerf/hook -m output_dynamic_mask/hook --eval --iterations 30000 --is_blender --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/dnerf/jumpingjacks -m output_dynamic_mask/jumpingjacks --eval --iterations 30000 --is_blender --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/dnerf/lego -m output_dynamic_mask/lego --eval --iterations 30000 --is_blender --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/dnerf/mutant -m output_dynamic_mask/mutant --eval --iterations 30000 --is_blender --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/dnerf/standup -m output_dynamic_mask/standup --eval --iterations 30000 --is_blender --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp
python train.py -s /root/datasets/dnerf/trex -m output_dynamic_mask/trex --eval --iterations 30000 --is_blender --use_velocity --use_dynamic_mask --dynamic_thresh_percentile 75 --velocity_network_type mlp

python render.py -m output_dynamic_mask/bouncingballs --mode render
python render.py -m output_dynamic_mask/hellwarrior --mode render
python render.py -m output_dynamic_mask/hook --mode render
python render.py -m output_dynamic_mask/jumpingjacks --mode render
python render.py -m output_dynamic_mask/lego --mode render
python render.py -m output_dynamic_mask/mutant --mode render
python render.py -m output_dynamic_mask/standup --mode render
python render.py -m output_dynamic_mask/trex --mode render

python metrics.py -m output_dynamic_mask/bouncingballs
python metrics.py -m output_dynamic_mask/hellwarrior
python metrics.py -m output_dynamic_mask/hook
python metrics.py -m output_dynamic_mask/jumpingjacks
python metrics.py -m output_dynamic_mask/lego
python metrics.py -m output_dynamic_mask/mutant
python metrics.py -m output_dynamic_mask/standup
python metrics.py -m output_dynamic_mask/trex

