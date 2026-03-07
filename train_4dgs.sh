# Neur3D Dataset
python train.py -s /root/datasets/Neu3D/coffee_martini -m output_4dgs/coffee_martini --eval --iterations 30000 --lazy_load --deform_type 4dgs
python train.py -s /root/datasets/Neu3D/cook_spinach -m output_4dgs/cook_spinach --eval --iterations 30000 --lazy_load --deform_type 4dgs
python train.py -s /root/datasets/Neu3D/cut_roasted_beef -m output_4dgs/cut_roasted_beef --eval --iterations 30000 --lazy_load --deform_type 4dgs
python train.py -s /root/datasets/Neu3D/flame_steak -m output_4dgs/flame_steak --eval --iterations 30000 --lazy_load --deform_type 4dgs
python train.py -s /root/datasets/Neu3D/sear_steak -m output_4dgs/sear_steak --eval --iterations 30000 --lazy_load --deform_type 4dgs
python train.py -s /root/datasets/Neu3D/flame_salmon_1 -m output_4dgs/flame_salmon_1 --eval --iterations 30000 --lazy_load --deform_type 4dgs

python render.py -m output_4dgs/coffee_martini --mode render --skip_train
python render.py -m output_4dgs/cook_spinach --mode render --skip_train
python render.py -m output_4dgs/cut_roasted_beef --mode render --skip_train
python render.py -m output_4dgs/flame_steak --mode render --skip_train
python render.py -m output_4dgs/sear_steak --mode render --skip_train
python render.py -m output_4dgs/flame_salmon_1 --mode render --skip_train

python metrics.py -m output_4dgs/coffee_martini
python metrics.py -m output_4dgs/cook_spinach
python metrics.py -m output_4dgs/cut_roasted_beef
python metrics.py -m output_4dgs/flame_steak
python metrics.py -m output_4dgs/sear_steak
python metrics.py -m output_4dgs/flame_salmon_1

