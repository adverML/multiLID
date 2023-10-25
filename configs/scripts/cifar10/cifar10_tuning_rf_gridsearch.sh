
python  detect.py --load_json cifar10/wrn28-10/k20/rf_fgsm_8255.json --tuning gridsearch --att fgsm
python  detect.py --load_json cifar10/wrn28-10/k20/rf_bim_8255.json  --tuning gridsearch --att bim
python  detect.py --load_json cifar10/wrn28-10/k20/rf_pgd_8255.json  --tuning gridsearch --att pgd
python  detect.py --load_json cifar10/wrn28-10/k20/rf_aa_8255.json   --tuning gridsearch --att aa
python  detect.py --load_json cifar10/wrn28-10/k20/rf_df.json   --tuning gridsearch --att df
python  detect.py --load_json cifar10/wrn28-10/k20/rf_cw.json   --tuning gridsearch --att cw