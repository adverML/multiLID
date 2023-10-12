# python gen.py --load_json imagenet/wrn50-2/bim_8255.json


python extract.py --load_json imagenet/wrn50-2/multilid/k30/bim_8255.json
python extract.py --load_json imagenet/wrn50-2/lid/k30/bim_8255.json


 python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_bim_8255.json