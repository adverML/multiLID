# python gen.py --load_json imagenet/wrn50-2/pgd_8255.json

# python gen.py --load_json imagenet/wrn50-2/pgd_4255.json
# python gen.py --load_json imagenet/wrn50-2/pgd_16255.json

python extract.py --load_json imagenet/wrn50-2/multilid/k30/pgd_8255.json
# python extract.py --load_json imagenet/wrn50-2/multilid/k30/pgd_16255.json

# python detect.py --load_json imagenet/wrn50-2/multilid/k30/lr_pgd_8255.json
# python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_pgd_8255.json

# python detect.py --load_json imagenet/wrn50-2/multilid/k30/lr_pgd_16255.json
# python detect.py --load_json imagenet/wrn50-2/multilid/k30/rf_pgd_16255.json



##### LID
python extract.py --load_json imagenet/wrn50-2/lid/k30/pgd_8255.json
# python extract.py --load_json imagenet/wrn50-2/lid/k30/pgd_16255.json
