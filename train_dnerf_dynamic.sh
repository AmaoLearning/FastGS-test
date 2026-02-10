# D-NeRF Dataset
python train.py -s /root/datasets/dnerf/bouncingballs -m output_dynamic_mask/bouncingballs_detach_0_50 --eval --iterations 30000 --is_blender --use_velocity --velocity_loss_percentile 0 --use_dynamic_mask --dynamic_thresh_percentile 50 --velocity_network_type mlp --detach_velocity_loss_from_deform
python train.py -s /root/datasets/dnerf/hellwarrior -m output_dynamic_mask/hellwarrior_detach_0_50 --eval --iterations 30000 --is_blender --use_velocity --velocity_loss_percentile 0 --use_dynamic_mask --dynamic_thresh_percentile 50 --velocity_network_type mlp --detach_velocity_loss_from_deform
python train.py -s /root/datasets/dnerf/hook -m output_dynamic_mask/hook_detach_0_50 --eval --iterations 30000 --is_blender --use_velocity --velocity_loss_percentile 0 --use_dynamic_mask --dynamic_thresh_percentile 50 --velocity_network_type mlp --detach_velocity_loss_from_deform
python train.py -s /root/datasets/dnerf/jumpingjacks -m output_dynamic_mask/jumpingjacks_detach_0_50 --eval --iterations 30000 --is_blender --use_velocity --velocity_loss_percentile 0 --use_dynamic_mask --dynamic_thresh_percentile 50 --velocity_network_type mlp --detach_velocity_loss_from_deform
python train.py -s /root/datasets/dnerf/lego -m output_dynamic_mask/lego_detach_0_50 --eval --iterations 30000 --is_blender --use_velocity --velocity_loss_percentile 0 --use_dynamic_mask --dynamic_thresh_percentile 50 --velocity_network_type mlp --detach_velocity_loss_from_deform
python train.py -s /root/datasets/dnerf/mutant -m output_dynamic_mask/mutant_detach_0_50 --eval --iterations 30000 --is_blender --use_velocity --velocity_loss_percentile 0 --use_dynamic_mask --dynamic_thresh_percentile 50 --velocity_network_type mlp --detach_velocity_loss_from_deform
python train.py -s /root/datasets/dnerf/standup -m output_dynamic_mask/standup_detach_0_50 --eval --iterations 30000 --is_blender --use_velocity --velocity_loss_percentile 0 --use_dynamic_mask --dynamic_thresh_percentile 50 --velocity_network_type mlp --detach_velocity_loss_from_deform
python train.py -s /root/datasets/dnerf/trex -m output_dynamic_mask/trex_detach_0_50 --eval --iterations 30000 --is_blender --use_velocity --velocity_loss_percentile 0 --use_dynamic_mask --dynamic_thresh_percentile 50 --velocity_network_type mlp --detach_velocity_loss_from_deform

python render.py -m output_dynamic_mask/bouncingballs_detach_0_50 --mode render
python render.py -m output_dynamic_mask/hellwarrior_detach_0_50 --mode render
python render.py -m output_dynamic_mask/hook_detach_0_50 --mode render
python render.py -m output_dynamic_mask/jumpingjacks_detach_0_50 --mode render
python render.py -m output_dynamic_mask/lego_detach_0_50 --mode render
python render.py -m output_dynamic_mask/mutant_detach_0_50 --mode render
python render.py -m output_dynamic_mask/standup_detach_0_50 --mode render
python render.py -m output_dynamic_mask/trex_detach_0_50 --mode render

python metrics.py -m output_dynamic_mask/bouncingballs_detach_0_50
python metrics.py -m output_dynamic_mask/hellwarrior_detach_0_50
python metrics.py -m output_dynamic_mask/hook_detach_0_50
python metrics.py -m output_dynamic_mask/jumpingjacks_detach_0_50
python metrics.py -m output_dynamic_mask/lego_detach_0_50
python metrics.py -m output_dynamic_mask/mutant_detach_0_50
python metrics.py -m output_dynamic_mask/standup_detach_0_50
python metrics.py -m output_dynamic_mask/trex_detach_0_50

