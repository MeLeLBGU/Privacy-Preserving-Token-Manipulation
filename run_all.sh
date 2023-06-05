#!/bin/bash

python main.py --save=2map_all.pt --remap=all --remap_count=2
python main.py --save=2map_all_shuffle.pt --shuffle --remap=all --remap_count=2
python main.py --save=2map_val.pt --remap_count=2 
python main.py --save=2map_val_shuffle.pt --remap_count=2 --shuffle 


python main.py --save=3map_all.pt --remap=all --remap_count=3
python main.py --save=3map_all_shuffle.pt --shuffle --remap=all --remap_count=3
python main.py --save=3map_val.pt --remap_count=3 
python main.py --save=3map_val_shuffle.pt --remap_count=3 --shuffle 


python main.py --save=4map_all.pt --remap=all --remap_count=4
python main.py --save=4map_all_shuffle.pt --shuffle --remap=all --remap_count=4
python main.py --save=4map_val.pt --remap_count=4 
python main.py --save=4map_val_shuffle.pt --remap_count=4 --shuffle 

