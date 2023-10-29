
python gen.py --load_json cifar10/wrn28-10/fgsm_8255_testset.json
python gen.py --load_json cifar10/wrn28-10/bim_8255_testset.json
python gen.py --load_json cifar10/wrn28-10/pgd_8255_testset.json
python gen.py --load_json cifar10/wrn28-10/aa_8255_testset.json
python gen.py --load_json cifar10/wrn28-10/df_testset.json
python gen.py --load_json cifar10/wrn28-10/cw_testset.json



python extract.py --load_json cifar10/wrn28-10/multilid/k20/fgsm_8255_testset.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/bim_8255_testset.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/pgd_8255_testset.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/aa_8255_testset.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/df_testset.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/cw_testset.json


