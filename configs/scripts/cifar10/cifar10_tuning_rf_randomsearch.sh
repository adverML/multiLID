
python  detect.py --load_json cifar10/wrn28-10/k20/rf_fgsm_8255.json --tuning randomsearch --att fgsm  --nr_samples 10000 --load_nor multilid_normalos_8255_testset.pt --load_adv multilid_adverlos_8255_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/rf_bim_8255.json  --tuning randomsearch --att bim   --nr_samples 10000 --load_nor multilid_normalos_8255_testset.pt --load_adv multilid_adverlos_8255_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/rf_pgd_8255.json  --tuning randomsearch --att pgd   --nr_samples 10000 --load_nor multilid_normalos_8255_testset.pt --load_adv multilid_adverlos_8255_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/rf_aa_8255.json   --tuning randomsearch --att aa    --nr_samples 10000 --load_nor multilid_normalos_8255_testset.pt --load_adv multilid_adverlos_8255_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/rf_df.json        --tuning randomsearch --att df    --nr_samples 10000 --load_nor multilid_normalos_testset.pt --load_adv multilid_adverlos_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/rf_cw.json        --tuning randomsearch --att cw    --nr_samples 10000 --load_nor multilid_normalos_testset.pt --load_adv multilid_adverlos_testset.pt
