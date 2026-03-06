# Neur3D Dataset — 使用形变场有限差分投影光流损失
python train.py -s /remote-home/panyicheng/datasets/Neu3D/coffee_martini -m output_flow/coffee_martini --eval --iterations 30000 --use_flow_loss --lazy_load --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/Neu3D/cook_spinach -m output_flow/cook_spinach --eval --iterations 30000 --use_flow_loss --lazy_load --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/Neu3D/cut_roasted_beef -m output_flow/cut_roasted_beef --eval --iterations 30000 --use_flow_loss --lazy_load --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/Neu3D/flame_steak -m output_flow/flame_steak --eval --iterations 30000 --use_flow_loss --lazy_load --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/Neu3D/sear_steak -m output_flow/sear_steak --eval --iterations 30000 --use_flow_loss --lazy_load --use_flow_mask
python train.py -s /remote-home/panyicheng/datasets/Neu3D/flame_salmon_1 -m output_flow/flame_salmon_1 --eval --iterations 30000 --use_flow_loss --lazy_load --use_flow_mask

python render.py -m output_flow/coffee_martini --mode render --skip_train
python render.py -m output_flow/cook_spinach --mode render --skip_train
python render.py -m output_flow/cut_roasted_beef --mode render --skip_train
python render.py -m output_flow/flame_steak --mode render --skip_train
python render.py -m output_flow/sear_steak --mode render --skip_train
python render.py -m output_flow/flame_salmon_1 --mode render --skip_train

python metrics.py -m output_flow/coffee_martini
python metrics.py -m output_flow/cook_spinach
python metrics.py -m output_flow/cut_roasted_beef
python metrics.py -m output_flow/flame_steak
python metrics.py -m output_flow/sear_steak
python metrics.py -m output_flow/flame_salmon_1

