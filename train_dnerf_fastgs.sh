# D-NeRF Dataset
python train.py -s /remote-home/panyicheng/datasets/dnerf/bouncingballs -m output/bouncingballs --eval --iterations 30000 --is_blender
python train.py -s /remote-home/panyicheng/datasets/dnerf/hellwarrior -m output/hellwarrior --eval --iterations 30000 --is_blender
python train.py -s /remote-home/panyicheng/datasets/dnerf/hook -m output/hook --eval --iterations 30000 --is_blender
python train.py -s /remote-home/panyicheng/datasets/dnerf/jumpingjacks -m output/jumpingjacks --eval --iterations 30000 --is_blender
python train.py -s /remote-home/panyicheng/datasets/dnerf/lego -m output/lego --eval --iterations 30000 --is_blender
python train.py -s /remote-home/panyicheng/datasets/dnerf/mutant -m output/mutant --eval --iterations 30000 --is_blender
python train.py -s /remote-home/panyicheng/datasets/dnerf/standup -m output/standup --eval --iterations 30000 --is_blender
python train.py -s /remote-home/panyicheng/datasets/dnerf/trex -m output/trex --eval --iterations 30000 --is_blender

python render.py -m output/bouncingballs --mode render
python render.py -m output/hellwarrior --mode render
python render.py -m output/hook --mode render
python render.py -m output/jumpingjacks --mode render
python render.py -m output/lego --mode render
python render.py -m output/mutant --mode render
python render.py -m output/standup --mode render
python render.py -m output/trex --mode render

python metrics.py -m output/bouncingballs
python metrics.py -m output/hellwarrior
python metrics.py -m output/hook
python metrics.py -m output/jumpingjacks
python metrics.py -m output/lego
python metrics.py -m output/mutant
python metrics.py -m output/standup
python metrics.py -m output/trex

