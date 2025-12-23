# Neur3D Dataset
python train.py -s /root/datasets/Neu3D/coffee_martini -m output/coffee_martini --eval --iterations 30000
python train.py -s /root/datasets/Neu3D/cook_spinach -m output/cook_spinach --eval --iterations 30000
python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output/cut_roasted_beef --eval --iterations 30000
python train.py -s /root/datasets/Neu3D/flame_steak -m output/flame_steak --eval --iterations 30000
python train.py -s /root/datasets/Neu3D/sear_steak -m output/sear_steak --eval --iterations 30000
python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output/flame_salmon_1 --eval --iterations 30000

python render.py -m output/coffee_martini --mode render
python render.py -m output/cook_spinach --mode render
python render.py -m output/cut_roasted_beef --mode render
python render.py -m output/flame_steak --mode render
python render.py -m output/sear_steak --mode render
python render.py -m output/flame_salmon_1 --mode render

python metrics.py -m output/coffee_martini
python metrics.py -m output/cook_spinach
python metrics.py -m output/cut_roasted_beef
python metrics.py -m output/flame_steak
python metrics.py -m output/sear_steak
python metrics.py -m output/flame_salmon_1

