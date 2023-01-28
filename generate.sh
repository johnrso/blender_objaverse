# create variables for the list of objects to query
python3 blender_objaverse.py --save_dir /shared/joso/objaverse_data_2 --num_samples 200 --cat shopping_bag --N 1000 --clear --lvis --tag dist &
python3 blender_objaverse.py --save_dir /shared/joso/objaverse_data_2 --num_samples 200 --cat shoulder_bag --N 1000 --clear --lvis --tag dist &
python3 blender_objaverse.py --save_dir /shared/joso/objaverse_data_2 --num_samples 200 --cat duffel_bag --N 1000 --clear --lvis --tag dist &
python3 blender_objaverse.py --save_dir /shared/joso/objaverse_data_2 --num_samples 200 --cat tote_bag --N 1000 --clear --lvis --tag dist &
# python3 blender_objaverse.py --save_dir /shared/joso/objaverse_data_2 --num_samples 200 --cat clutch_bag --N 1000 --clear --lvis --tag dist &
# python3 blender_objaverse.py --save_dir /shared/joso/objaverse_data_2 --num_samples 200 --cat grocery_bag --N 1000 --clear --lvis --tag dist &
# python3 blender_objaverse.py --save_dir /shared/joso/objaverse_data_2 --num_samples 200 --cat plastic_bag --N 1000 --clear --lvis --tag dist &
