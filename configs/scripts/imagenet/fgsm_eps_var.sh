python gen.py --load_json imagenet/wrn50-2/fgsm_0.001.json
python gen.py --load_json imagenet/wrn50-2/fgsm_0.01.json
python gen.py --load_json imagenet/wrn50-2/fgsm_0.1.json
python gen.py --load_json imagenet/wrn50-2/fgsm_1.0.json

python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_0.001.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_0.01.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_0.1.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_1.0.json

python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_0.001.json
python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_0.01.json
python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_0.1.json
python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_fgsm_1.0.json

# attack: fgsm
# eps     ASR (%)  AUC (%)
# 0.001   42.48    71.57
# 0.01    41.90    73.15
# 0.1     41.75    70.56
# 1       42.70    78.18

# attack: fgsm
# eps     ASR (%)  AUC (%)
# 0.001   25.10    52.14
# 0.01    43.98    60.44
# 0.1     43.52    99.16
# 1       99.95    100