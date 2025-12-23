# Neur3D Dataset
python train.py -s /root/datasets/Neu3D/coffee_martini -m output_velocity/coffee_martini --eval --iterations 30000 --use_velocity
python train.py -s /root/datasets/Neu3D/cook_spinach -m output_velocity/cook_spinach --eval --iterations 30000 --use_velocity
python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_velocity/cut_roasted_beef --eval --iterations 30000 --use_velocity
python train.py -s /root/datasets/Neu3D/flame_steak -m output_velocity/flame_steak --eval --iterations 30000 --use_velocity
python train.py -s /root/datasets/Neu3D/sear_steak -m output_velocity/sear_steak --eval --iterations 30000 --use_velocity
python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_velocity/flame_salmon_1 --eval --iterations 30000 --use_velocity

python render.py -m output_velocity/coffee_martini --mode render
python render.py -m output_velocity/cook_spinach --mode render
python render.py -m output_velocity/cut_roasted_beef --mode render
python render.py -m output_velocity/flame_steak --mode render
python render.py -m output_velocity/sear_steak --mode render
python render.py -m output_velocity/flame_salmon_1 --mode render

python metrics.py -m output_velocity/coffee_martini
python metrics.py -m output_velocity/cook_spinach
python metrics.py -m output_velocity/cut_roasted_beef
python metrics.py -m output_velocity/flame_steak
python metrics.py -m output_velocity/sear_steak
python metrics.py -m output_velocity/flame_salmon_1

