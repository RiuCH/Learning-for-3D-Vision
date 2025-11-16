
# # 1
# python volume_rendering_main.py --config-name=box

# # 2
# python volume_rendering_main.py --config-name=train_box

# # 3
# python volume_rendering_main.py --config-name=nerf_lego

# # 4.1 - use_view_directions: True
# python volume_rendering_main.py --config-name=nerf_materials_highres

# #  4.2 - fine_network: True
# python volume_rendering_main.py --config-name=nerf_lego

# # 5
# python -m surface_rendering_main --config-name=torus_surface

# # 6 - num_epochs: 10000
# python -m surface_rendering_main --config-name=points_surface

# # 7 
python -m surface_rendering_main --config-name=volsdf_surface
python -m surface_rendering_main --config-name=volsdf_surface1
python -m surface_rendering_main --config-name=volsdf_surface2
python -m surface_rendering_main --config-name=volsdf_surface3
python -m surface_rendering_main --config-name=volsdf_surface4


# 8.1
python -m surface_rendering_main --config-name=combined_surface

# 8.2 limit_view: 30
# python volume_rendering_main.py --config-name=nerf_lego
# python -m surface_rendering_main --config-name=volsdf_surface

# 8.3 use_neus: True,  s: 10
python -m surface_rendering_main --config-name=volsdf_surface