# NeRF-DS Dataset
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_as -m output/vrig_as --eval --iterations 30000
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_basin -m output/vrig_basin --eval --iterations 30000
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_bell -m output/vrig_bell --eval --iterations 30000
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_cup -m output/vrig_cup --eval --iterations 30000
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_plate -m output/vrig_plate --eval --iterations 30000
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_press -m output/vrig_press --eval --iterations 30000
python train.py -s /remote-home/panyicheng/datasets/nerf-ds/vrig_sieve -m output/vrig_sieve --eval --iterations 30000

python render.py -m output/vrig_as --mode render
python render.py -m output/vrig_basin --mode render
python render.py -m output/vrig_bell --mode render
python render.py -m output/vrig_cup --mode render
python render.py -m output/vrig_plate --mode render
python render.py -m output/vrig_press --mode render
python render.py -m output/vrig_sieve --mode render

python metrics.py -m output/vrig_as
python metrics.py -m output/vrig_basin
python metrics.py -m output/vrig_bell
python metrics.py -m output/vrig_cup
python metrics.py -m output/vrig_plate
python metrics.py -m output/vrig_press
python metrics.py -m output/vrig_sieve

