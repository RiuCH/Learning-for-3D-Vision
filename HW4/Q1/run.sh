
python render.py
python train.py
python render.py --out_path ./output_sh 

python train_harder_scene_baseline.py --out_path ./output_baseline --data_path ./data/materials --gaussians_per_splat 2048

python train_harder_scene.py --out_path ./output_hard --data_path ./data/materials --gaussians_per_splat 2048 --num_itrs 10000
