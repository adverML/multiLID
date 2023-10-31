
python  detect.py --load_json cifar10/wrn28-10/k20/lr_fgsm_8255.json --tuning gridsearch --att fgsm  --nr_samples 10000 --load_nor multilid_normalos_8255_testset.pt --load_adv multilid_adverlos_8255_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/lr_bim_8255.json  --tuning gridsearch --att bim   --nr_samples 10000 --load_nor multilid_normalos_8255_testset.pt --load_adv multilid_adverlos_8255_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/lr_pgd_8255.json  --tuning gridsearch --att pgd   --nr_samples 10000 --load_nor multilid_normalos_8255_testset.pt --load_adv multilid_adverlos_8255_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/lr_aa_8255.json   --tuning gridsearch --att aa    --nr_samples 10000 --load_nor multilid_normalos_8255_testset.pt --load_adv multilid_adverlos_8255_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/lr_df.json        --tuning gridsearch --att df    --nr_samples 10000 --load_nor multilid_normalos_testset.pt --load_adv multilid_adverlos_testset.pt
python  detect.py --load_json cifar10/wrn28-10/k20/lr_cw.json        --tuning gridsearch --att cw    --nr_samples 10000 --load_nor multilid_normalos_testset.pt --load_adv multilid_adverlos_testset.pt


  