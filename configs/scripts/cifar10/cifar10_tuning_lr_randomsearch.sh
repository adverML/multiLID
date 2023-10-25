python  detect.py --load_json cifar10/wrn28-10/k20/lr_fgsm_8255.json --tuning randomsearch --att fgsm
python  detect.py --load_json cifar10/wrn28-10/k20/lr_bim_8255.json  --tuning randomsearch --att bim
python  detect.py --load_json cifar10/wrn28-10/k20/lr_pgd_8255.json  --tuning randomsearch --att pgd
python  detect.py --load_json cifar10/wrn28-10/k20/lr_aa_8255.json   --tuning randomsearch --att aa
python  detect.py --load_json cifar10/wrn28-10/k20/lr_df.json   --tuning randomsearch --att df
python  detect.py --load_json cifar10/wrn28-10/k20/lr_cw.json   --tuning randomsearch --att cw
