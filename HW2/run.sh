
# Question 1
python fit_data.py --type 'vox' --max_iter 30000

python fit_data.py --type 'point' --max_iter 15000

python fit_data.py --type 'mesh' 

# Question 2

# 2.1
python train_model.py --type 'vox' --batch_size 256 --max_iter 2000 --save_freq 1000

# 2.2
python train_model.py --type 'point' --batch_size 256 --max_iter 10000

# 2.3
python train_model.py --type 'mesh' --batch_size 256 --max_iter 10000

# 2.4
python eval_model.py --type 'vox' --load_checkpoint --vis_freq 150
python eval_model.py --type 'point' --load_checkpoint --vis_freq 150
python eval_model.py --type 'mesh' --load_checkpoint --vis_freq 150


# 2.5
python train_model.py --type 'point' --batch_size 256 --max_iter 10000 --n_points 500
python train_model.py --type 'point' --batch_size 256 --max_iter 10000 --n_points 2000

python eval_model.py --type 'point' --load_checkpoint --vis_freq 150 --n_points 500
python eval_model.py --type 'point' --load_checkpoint --vis_freq 150 --n_points 2000

# 2.6
python eval_model.py --type 'vox' --load_checkpoint --vis_freq 150 --interpret

# Question 3

# 3.1
python train_model.py --type 'implicit' --batch_size 8 --max_iter 2010

## 3.2
python train_model.py --type 'parametric' --max_iter 10000 --n_points 2000
python eval_model.py --type 'parametric' --load_checkpoint --vis_freq 150 --n_points 5000

# 3.3
python train_model.py --type 'point' --batch_size 256 --max_iter 10000