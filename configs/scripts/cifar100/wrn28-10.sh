
export run="run_3"

# python gen.py --load_json cifar100/wrn28-10/fgsm_8255.json
# python gen.py --load_json cifar100/wrn28-10/bim_8255.json
# python gen.py --load_json cifar100/wrn28-10/pgd_8255.json
# python gen.py --load_json cifar100/wrn28-10/aa_8255.json
# python gen.py --load_json cifar100/wrn28-10/cw.json
# python gen.py --load_json cifar100/wrn28-10/df.json

``
python extract.py --load_json cifar100/wrn28-10/multilid/k20/fgsm_8255.json --run_nr $run 
python extract.py --load_json cifar100/wrn28-10/multilid/k20/bim_8255.json  --run_nr $run  
python extract.py --load_json cifar100/wrn28-10/multilid/k20/pgd_8255.json  --run_nr $run  
python extract.py --load_json cifar100/wrn28-10/multilid/k20/aa_8255.json   --run_nr $run  
python extract.py --load_json cifar100/wrn28-10/multilid/k20/df.json        --run_nr $run  
python extract.py --load_json cifar100/wrn28-10/multilid/k20/cw.json        --run_nr $run  


# python extract.py --load_json cifar100/wrn28-10/lid/k20/fgsm_8255.json --run_nr $run  
# python extract.py --load_json cifar100/wrn28-10/lid/k20/bim_8255.json  --run_nr $run  
# python extract.py --load_json cifar100/wrn28-10/lid/k20/pgd_8255.json  --run_nr $run  
# python extract.py --load_json cifar100/wrn28-10/lid/k20/aa_8255.json   --run_nr $run  
# python extract.py --load_json cifar100/wrn28-10/lid/k20/df.json        --run_nr $run  
# python extract.py --load_json cifar100/wrn28-10/lid/k20/cw.json        --run_nr $run 


