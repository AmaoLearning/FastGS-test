# HyperNeRF Dataset — 使用形变场有限差分投影光流损失
python train.py -s /remote-home/panyicheng/datasets/hypernerf/interp_espresso -m output_flow/interp_espresso --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/hypernerf/interp_americano -m output_flow/interp_americano --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/hypernerf/interp_split-cookie -m output_flow/interp_split-cookie --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/hypernerf/vrig-chicken -m output_flow/vrig-chicken --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/hypernerf/interp_torchocolate -m output_flow/interp_torchocolate --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/hypernerf/interp_cut-lemon1 -m output_flow/interp_cut-lemon1 --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/hypernerf/interp_hand1-dense-v2 -m output_flow/interp_hand1-dense-v2 --eval --iterations 30000 --use_flow_loss
python train.py -s /remote-home/panyicheng/datasets/hypernerf/vrig-3dprinter -m output_flow/vrig-3dprinter --eval --iterations 30000 --use_flow_loss

python render.py -m output_flow/interp_espresso --mode render
python render.py -m output_flow/interp_americano --mode render
python render.py -m output_flow/interp_split-cookie --mode render
python render.py -m output_flow/vrig-chicken --mode render
python render.py -m output_flow/interp_torchocolate --mode render
python render.py -m output_flow/interp_cut-lemon1 --mode render
python render.py -m output_flow/interp_hand1-dense-v2 --mode render
python render.py -m output_flow/vrig-3dprinter --mode render

python metrics.py -m output_flow/interp_espresso
python metrics.py -m output_flow/interp_americano
python metrics.py -m output_flow/interp_split-cookie
python metrics.py -m output_flow/vrig-chicken
python metrics.py -m output_flow/interp_torchocolate
python metrics.py -m output_flow/interp_cut-lemon1
python metrics.py -m output_flow/interp_hand1-dense-v2
python metrics.py -m output_flow/vrig-3dprinter

