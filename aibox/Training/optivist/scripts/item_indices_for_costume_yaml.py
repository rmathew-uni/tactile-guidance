from pycocotools.coco import COCO
import requests
import sys
from tqdm import tqdm

# instantiate COCO specifying the annotations json path
coco = COCO('/share/neurobiopsychologie/datasets/coco_subset_optivist/instances_train2017.json')

items_list = [['potted plant'], ['clock'], ['bicycle'], ['wine glass'],['bottle'],['apple'],['orange'],['banana'],['bowl'],['cell phone'],['spoon'],['zebra'],['knife'],['fork'],['car'],['motorcycle'],['cup'],['pizza'],['scissors'],['toothbrush']]

lst_cat = []
items_dict = {}
for item in items_list:
    # index of category (e.g banana is 52)
    catIds = coco.getCatIds(catNms=item)
    lst_cat.append(catIds)
    
    # dictionary with items and category indices 
    items_dict[str(item)] = catIds

lst_cat.sort()


# function for getting a key for specific value
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


# iterating through 
for i,v in enumerate(lst_cat):
    obj = get_keys_from_value(items_dict,v)
    result = obj[0].strip("[']").strip("'")
    print(f'{i}: {result}')