# D-NeRF Dataset — 使用形变场有限差分投影光流损失
python train.py -s /remote-home/panyicheng/datasets/dnerf/bouncingballs -m output_flow/bouncingballs --eval --iterations 30000 --is_blender --use_flow_loss --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/dnerf/hellwarrior -m output_flow/hellwarrior --eval --iterations 30000 --is_blender --use_flow_loss --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/dnerf/hook -m output_flow/hook --eval --iterations 30000 --is_blender --use_flow_loss --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/dnerf/jumpingjacks -m output_flow/jumpingjacks --eval --iterations 30000 --is_blender --use_flow_loss --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/dnerf/lego -m output_flow/lego --eval --iterations 30000 --is_blender --use_flow_loss --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/dnerf/mutant -m output_flow/mutant --eval --iterations 30000 --is_blender --use_flow_loss --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/dnerf/standup -m output_flow/standup --eval --iterations 30000 --is_blender --use_flow_loss --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/dnerf/trex -m output_flow/trex --eval --iterations 30000 --is_blender --use_flow_loss --use_flow_mask

python render.py -m output_flow/bouncingballs --mode render
python render.py -m output_flow/hellwarrior --mode render
python render.py -m output_flow/hook --mode render
python render.py -m output_flow/jumpingjacks --mode render
python render.py -m output_flow/lego --mode render
python render.py -m output_flow/mutant --mode render
python render.py -m output_flow/standup --mode render
python render.py -m output_flow/trex --mode render

python metrics.py -m output_flow/bouncingballs
python metrics.py -m output_flow/hellwarrior
python metrics.py -m output_flow/hook
python metrics.py -m output_flow/jumpingjacks
python metrics.py -m output_flow/lego
python metrics.py -m output_flow/mutant
python metrics.py -m output_flow/standup
python metrics.py -m output_flow/trex

