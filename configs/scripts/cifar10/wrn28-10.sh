python gen.py --load_json cifar10/wrn28-10/fgsm_8255.json
python gen.py --load_json cifar10/wrn28-10/bim_8255.json
python gen.py --load_json cifar10/wrn28-10/pgd_8255.json
python gen.py --load_json cifar10/wrn28-10/cw.json
python gen.py --load_json cifar10/wrn28-10/df.json



python extract.py --load_json cifar10/wrn28-10/multilid/k30/fgsm_8255.json
python extract.py --load_json cifar10/wrn28-10/multilid/k30/bim_8255.json
python extract.py --load_json cifar10/wrn28-10/multilid/k30/pgd_8255.json
python extract.py --load_json cifar10/wrn28-10/multilid/k30/aa_8255.json
python extract.py --load_json cifar10/wrn28-10/multilid/k30/df.json
python extract.py --load_json cifar10/wrn28-10/multilid/k30/cw.json


python extract.py --load_json cifar10/wrn28-10/lid/k30/fgsm_8255.json
python extract.py --load_json cifar10/wrn28-10/lid/k30/bim_8255.json
python extract.py --load_json cifar10/wrn28-10/lid/k30/pgd_8255.json
python extract.py --load_json cifar10/wrn28-10/lid/k30/aa_8255.json
python extract.py --load_json cifar10/wrn28-10/lid/k30/df.json
python extract.py --load_json cifar10/wrn28-10/lid/k30/cw.json