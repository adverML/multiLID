# python gen.py --load_json pgd/pgd_8255.json

# python gen.py --load_json pgd/pgd_4255.json
# python gen.py --load_json pgd/pgd_16255.json

python extract.py --load_json multilid/k30/pgd_8255.json
# python extract.py --load_json multilid/k30/pgd_16255.json

# python detect.py --load_json multilid/k30/lr_pgd_8255.json
# python detect.py --load_json multilid/k30/rf_pgd_8255.json

# python detect.py --load_json multilid/k30/lr_pgd_16255.json
# python detect.py --load_json multilid/k30/rf_pgd_16255.json



##### LID
python extract.py --load_json lid/k30/pgd_8255.json
# python extract.py --load_json lid/k30/pgd_16255.json
