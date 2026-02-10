# Neur3D Dataset
python train.py -s /root/datasets/hypernerf/interp_espresso -m output/interp_espresso --eval --iterations 30000
python train.py -s /root/datasets/hypernerf/interp_americano -m output/interp_americano --eval --iterations 30000
python train.py -s /root/datasets/hypernerf/interp_split-cookie -m output/interp_split-cookie --eval --iterations 30000
python train.py -s /root/datasets/hypernerf/vrig-chicken -m output/vrig-chicken --eval --iterations 30000
python train.py -s /root/datasets/hypernerf/interp_torchocolate -m output/interp_torchocolate --eval --iterations 30000
python train.py -s /root/datasets/hypernerf/interp_cut-lemon1 -m output/interp_cut-lemon --eval --iterations 30000
python train.py -s /root/datasets/hypernerf/interp_hand1-dense-v2 -m output/interp_hand1-dense-v2 --eval --iterations 30000
python train.py -s /root/datasets/hypernerf/vrig-3dprinter -m output/vrig-3dprinter --eval --iterations 30000

python render.py -m output/interp_espresso --mode render
python render.py -m output/interp_americano --mode render
python render.py -m output/interp_split-cookie --mode render
python render.py -m output/vrig-chicken --mode render
python render.py -m output/interp_torchocolate --mode render
python render.py -m output/interp_cut-lemon1 --mode render
python render.py -m output/interp_hand1-dense-v2 --mode render
python render.py -m output/vrig-3dprinter --mode render

python metrics.py -m output/interp_espresso
python metrics.py -m output/interp_americano
python metrics.py -m output/interp_split-cookie
python metrics.py -m output/vrig-chicken
python metrics.py -m output/interp_torchocolate
python metrics.py -m output/interp_cut-lemon1
python metrics.py -m output/interp_hand1-dense-v2
python metrics.py -m output/vrig-3dprinter

