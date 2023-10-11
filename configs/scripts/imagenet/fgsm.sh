# python gen.py --load_json imagenet/wrn50-2/fgsm/fgsm_8255.json
# python gen.py --load_json imagenet/wrn50-2/fgsm/fgsm_16255.json
# python gen.py --load_json imagenet/wrn50-2/fgsm/fgsm_32255.json

# python extract.py --load_json multilid/k30/fgsm_4255.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_8255.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_16255.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_32255.json

# python detect.py --load_json multilid/k30/lr_fgsm_8255.json
# python detect.py --load_json multilid/k30/rf_fgsm_8255.json

# python detect.py --load_json multilid/k30/lr_fgsm_16255.json
# python detect.py --load_json multilid/k30/rf_fgsm_16255.json


##### LID
python extract.py --load_json imagenet/wrn50-2/lid/k30/fgsm_8255.json
python extract.py --load_json imagenet/wrn50-2/lid/k30/fgsm_16255.json
python extract.py --load_json imagenet/wrn50-2/lid/k30/fgsm_32255.json

python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_8255.json


python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_0.001.json
python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_0.01.json
python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_0.1.json
python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_1.0.json
