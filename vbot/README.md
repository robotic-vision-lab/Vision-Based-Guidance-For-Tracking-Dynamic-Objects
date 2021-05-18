## VBOT Repository

This directory contains main code files with tests, experiments and implementations

### File structure

📦vbot  
 ┣ 📂algorithms  
 ┃ ┣ 📂feature_detection  
 ┃ ┃ ┃ ┣ 📜sift.cpython-37.pyc  
 ┃ ┃ ┃ ┗ 📜__init__.cpython-37.pyc  
 ┃ ┃ ┣ 📜sift.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂feature_match  
 ┃ ┃ ┃ ┣ 📜brute_l2.cpython-37.pyc  
 ┃ ┃ ┃ ┗ 📜__init__.cpython-37.pyc  
 ┃ ┃ ┣ 📜brute_l2.py  
 ┃ ┃ ┣ 📜flann.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂optical_flow  
 ┃ ┃ ┣ 📂results  
 ┃ ┃ ┃ ┣ 📜optical_flow_farn_car.jpg  
 ┃ ┃ ┃ ┣ 📜optical_flow_farn_dimetrodon.jpg  
 ┃ ┃ ┃ ┣ 📜optical_flow_farn_rubber.jpg  
 ┃ ┃ ┃ ┣ 📜optical_flow_farn_synth.jpg  
 ┃ ┃ ┃ ┣ 📜optical_flow_HS_car.jpg  
 ┃ ┃ ┃ ┣ 📜optical_flow_HS_dimetrodon.jpg  
 ┃ ┃ ┃ ┣ 📜optical_flow_HS_rubber.jpg  
 ┃ ┃ ┃ ┗ 📜optical_flow_HS_synth.jpg  
 ┃ ┃ ┣ 📜farneback.py  
 ┃ ┃ ┣ 📜horn_schunk.py  
 ┃ ┃ ┣ 📜lucas_kanade.py  
 ┃ ┃ ┣ 📜README.md  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂template_match  
 ┃ ┃ ┃ ┣ 📜corr_coeff_norm.cpython-37.pyc  
 ┃ ┃ ┃ ┗ 📜__init__.cpython-37.pyc  
 ┃ ┃ ┣ 📜corr_coeff_norm.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┃ ┗ 📜__init__.cpython-37.pyc  
 ┃ ┣ 📜README.md  
 ┃ ┗ 📜__init__.py  
 ┣ 📂experiments  
 ┃ ┣ 📂assets  
 ┃ ┃ ┣ 📜alienGreen_badge2.png  
 ┃ ┃ ┣ 📜car.png  
 ┃ ┃ ┣ 📜car3.png  
 ┃ ┃ ┣ 📜cars_1.png  
 ┃ ┃ ┣ 📜cars_2.png  
 ┃ ┃ ┣ 📜cars_3.png  
 ┃ ┃ ┗ 📜cross_hair2.png  
 ┃ ┣ 📂exp  
 ┃ ┃ ┣ 📂tmp  
 ┃ ┃ ┃ ┗ 📂track_tmp  
 ┃ ┃ ┣ 📜bar.py  
 ┃ ┃ ┣ 📜block.py  
 ┃ ┃ ┣ 📜car.py  
 ┃ ┃ ┣ 📜colors.py  
 ┃ ┃ ┣ 📜controller.py  
 ┃ ┃ ┣ 📜drone_camera.py  
 ┃ ┃ ┣ 📜ekf.py  
 ┃ ┃ ┣ 📜game_utils.py  
 ┃ ┃ ┣ 📜high_precision_clock.py  
 ┃ ┃ ┣ 📜manager.py  
 ┃ ┃ ┣ 📜mother_tracker.py  
 ┃ ┃ ┣ 📜my_imports.py  
 ┃ ┃ ┣ 📜optical_flow_config.py  
 ┃ ┃ ┣ 📜settings.py  
 ┃ ┃ ┣ 📜simulator.py  
 ┃ ┃ ┣ 📜target.py  
 ┃ ┃ ┣ 📜tracker.py  
 ┃ ┃ ┣ 📜__init__.py  
 ┃ ┃ ┗ 📜__main__.py  
 ┃ ┣ 📂exp_2  
 ┃ ┃ ┣ 📂tmp  
 ┃ ┃ ┃ ┗ 📂track_tmp  
 ┃ ┃ ┣ 📜bar.py  
 ┃ ┃ ┣ 📜block.py  
 ┃ ┃ ┣ 📜car.py  
 ┃ ┃ ┣ 📜colors.py  
 ┃ ┃ ┣ 📜controller.py  
 ┃ ┃ ┣ 📜drone_camera.py  
 ┃ ┃ ┣ 📜ekf.py  
 ┃ ┃ ┣ 📜game_utils.py  
 ┃ ┃ ┣ 📜high_precision_clock.py  
 ┃ ┃ ┣ 📜manager.py  
 ┃ ┃ ┣ 📜my_imports.py  
 ┃ ┃ ┣ 📜optical_flow_config.py  
 ┃ ┃ ┣ 📜settings.py  
 ┃ ┃ ┣ 📜simulator.py  
 ┃ ┃ ┣ 📜target.py  
 ┃ ┃ ┣ 📜tracker.py  
 ┃ ┃ ┣ 📜__init__.py  
 ┃ ┃ ┗ 📜__main__.py  
 ┃ ┣ 📂exp_lc  
 ┃ ┃ ┣ 📂tmp  
 ┃ ┃ ┃ ┗ 📂track_tmp  
 ┃ ┃ ┣ 📜bar.py  
 ┃ ┃ ┣ 📜block.py  
 ┃ ┃ ┣ 📜car.py  
 ┃ ┃ ┣ 📜colors.py  
 ┃ ┃ ┣ 📜controller.py  
 ┃ ┃ ┣ 📜drone_camera.py  
 ┃ ┃ ┣ 📜ekf.py  
 ┃ ┃ ┣ 📜game_utils.py  
 ┃ ┃ ┣ 📜high_precision_clock.py  
 ┃ ┃ ┣ 📜kf.py  
 ┃ ┃ ┣ 📜maf.py  
 ┃ ┃ ┣ 📜manager.py  
 ┃ ┃ ┣ 📜my_imports.py  
 ┃ ┃ ┣ 📜optical_flow_config.py  
 ┃ ┃ ┣ 📜settings.py  
 ┃ ┃ ┣ 📜simulator.py  
 ┃ ┃ ┣ 📜tracker.py  
 ┃ ┃ ┣ 📜__init__.py  
 ┃ ┃ ┗ 📜__main__.py  
 ┃ ┣ 📂exp_occ  
 ┃ ┃ ┣ 📂tmp  
 ┃ ┃ ┃ ┗ 📂track_tmp  
 ┃ ┃ ┣ 📜bar.py  
 ┃ ┃ ┣ 📜block.py  
 ┃ ┃ ┣ 📜car.py  
 ┃ ┃ ┣ 📜colors.py  
 ┃ ┃ ┣ 📜controller.py  
 ┃ ┃ ┣ 📜drone_camera.py  
 ┃ ┃ ┣ 📜ekf.py  
 ┃ ┃ ┣ 📜game_utils.py  
 ┃ ┃ ┣ 📜high_precision_clock.py  
 ┃ ┃ ┣ 📜kf.py  
 ┃ ┃ ┣ 📜maf.py  
 ┃ ┃ ┣ 📜manager.py  
 ┃ ┃ ┣ 📜my_imports.py  
 ┃ ┃ ┣ 📜optical_flow_config.py  
 ┃ ┃ ┣ 📜settings.py  
 ┃ ┃ ┣ 📜simulator.py  
 ┃ ┃ ┣ 📜tracker.py  
 ┃ ┃ ┣ 📜__init__.py  
 ┃ ┃ ┗ 📜__main__.py  
 ┃ ┣ 📂exp_sf  
 ┃ ┃ ┣ 📂tmp  
 ┃ ┃ ┃ ┗ 📂track_tmp  
 ┃ ┃ ┣ 📜bar.py  
 ┃ ┃ ┣ 📜block.py  
 ┃ ┃ ┣ 📜car.py  
 ┃ ┃ ┣ 📜colors.py  
 ┃ ┃ ┣ 📜controller.py  
 ┃ ┃ ┣ 📜drone_camera.py  
 ┃ ┃ ┣ 📜ekf.py  
 ┃ ┃ ┣ 📜game_utils.py  
 ┃ ┃ ┣ 📜high_precision_clock.py  
 ┃ ┃ ┣ 📜kf.py  
 ┃ ┃ ┣ 📜maf.py  
 ┃ ┃ ┣ 📜manager.py  
 ┃ ┃ ┣ 📜my_imports.py  
 ┃ ┃ ┣ 📜optical_flow_config.py  
 ┃ ┃ ┣ 📜settings.py  
 ┃ ┃ ┣ 📜simulator.py  
 ┃ ┃ ┣ 📜tracker.py  
 ┃ ┃ ┣ 📜__init__.py  
 ┃ ┃ ┗ 📜__main__.py  
 ┃ ┣ 📂sim_outputs  
 ┃ ┣ 📂tmp  
 ┃ ┃ ┣ 📂track_tmp  
 ┃ ┣ 📜plot_info.csv  
 ┃ ┗ 📜__init__.py  
 ┣ 📂logs  
 ┃ ┗ 📜debug.log  
 ┣ 📂notebooks  
 ┃ ┣ 📜debug.log  
 ┃ ┣ 📜test_OF_farneback.ipynb  
 ┃ ┣ 📜test_OF_HS.ipynb  
 ┃ ┣ 📜test_OF_LK.ipynb  
 ┃ ┗ 📜__init__.py  
 ┣ 📂utils  
 ┃ ┣ 📜data_synth_utils.py  
 ┃ ┣ 📜img_utils.py  
 ┃ ┣ 📜optical_flow_utils.py  
 ┃ ┣ 📜vid_utils.py  
 ┃ ┣ 📜window_utils.py  
 ┃ ┗ 📜__init__.py  
 ┣ 📜.pylintrc  
 ┣ 📜README.md  
 ┗ 📜__init__.py  