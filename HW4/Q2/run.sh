

python Q21_image_optimization.py --prompt "a hamburger" --sds_guidance 0 --postfix 0 
python Q21_image_optimization.py --prompt "a hamburger" --sds_guidance 1 --postfix 1 

python Q21_image_optimization.py --prompt "a standing corgi dog" --sds_guidance 0 --postfix 0 
python Q21_image_optimization.py --prompt "a standing corgi dog" --sds_guidance 1 --postfix 1 

python Q21_image_optimization.py --prompt "a gorilla wearing suit with sunglasses in minecraft theme" --sds_guidance 0 --postfix 0 
python Q21_image_optimization.py --prompt "a gorilla wearing suit with sunglasses in minecraft theme" --sds_guidance 1 --postfix 1 

python Q21_image_optimization.py --prompt "a orange shubby cat with a black strip holding sword" --sds_guidance 0 --postfix 0 
python Q21_image_optimization.py --prompt "a orange shubby cat with a black strip holding sword" --sds_guidance 1 --postfix 1 


python Q22_mesh_optimization.py --prompt "a pink and yellow stripe cow"
python Q22_mesh_optimization.py --prompt "a black and white cow"  


export CUDA_VISIBLE_DEVICES=2 && python Q23_nerf_optimization.py --prompt "a computer" --postfix 0 &
export CUDA_VISIBLE_DEVICES=3 && python Q23_nerf_optimization.py --prompt "a computer" --view_dep_text 1 --postfix 1 

export CUDA_VISIBLE_DEVICES=4 && python Q23_nerf_optimization.py --prompt "a hotdog" --postfix 0 &
export CUDA_VISIBLE_DEVICES=5 && python Q23_nerf_optimization.py --prompt "a hotdog" --view_dep_text 1 --postfix 1 &

export CUDA_VISIBLE_DEVICES=6 && python Q23_nerf_optimization.py --prompt "a standing shiba dog" --postfix 0 &
export CUDA_VISIBLE_DEVICES=7 && python Q23_nerf_optimization.py --prompt "a standing shiba dog" --view_dep_text 1 --postfix 1 