# NeRF-DS Dataset — 使用形变场有限差分投影光流损失
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_as -m output_flow/vrig_as --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_basin -m output_flow/vrig_basin --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_bell -m output_flow/vrig_bell --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_cup -m output_flow/vrig_cup --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_plate -m output_flow/vrig_plate --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_press -m output_flow/vrig_press --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_sieve -m output_flow/vrig_sieve --eval --iterations 30000 --use_flow_loss

python render.py -m output_flow/vrig_as --mode render
python render.py -m output_flow/vrig_basin --mode render
python render.py -m output_flow/vrig_bell --mode render
python render.py -m output_flow/vrig_cup --mode render
python render.py -m output_flow/vrig_plate --mode render
python render.py -m output_flow/vrig_press --mode render
python render.py -m output_flow/vrig_sieve --mode render

python metrics.py -m output_flow/vrig_as
python metrics.py -m output_flow/vrig_basin
python metrics.py -m output_flow/vrig_bell
python metrics.py -m output_flow/vrig_cup
python metrics.py -m output_flow/vrig_plate
python metrics.py -m output_flow/vrig_press
python metrics.py -m output_flow/vrig_sieve

