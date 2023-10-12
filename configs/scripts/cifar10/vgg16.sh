

# python gen.py --load_json cifar10/vgg16/fgsm_8255.json
python gen.py --load_json cifar10/vgg16/bim_8255.json
python gen.py --load_json cifar10/vgg16/pgd_8255.json
python gen.py --load_json cifar10/vgg16/aa_8255.json
python gen.py --load_json cifar10/vgg16/df.json
python gen.py --load_json cifar10/vgg16/cw.json


# # multiLID
python extract.py --load_json cifar10/vgg16/multilid/k20/fgsm_8255.json
python extract.py --load_json cifar10/vgg16/multilid/k20/bim_8255.json
python extract.py --load_json cifar10/vgg16/multilid/k20/pgd_8255.json
python extract.py --load_json cifar10/vgg16/multilid/k20/aa_8255.json
python extract.py --load_json cifar10/vgg16/multilid/k20/df.json
python extract.py --load_json cifar10/vgg16/multilid/k20/cw.json


# # python detect.py --load_json cifar10/vgg16/multilid/k20/fgsm_8255.json



# # LID
python extract.py --load_json cifar10/vgg16/lid/k20/fgsm_8255.json
python extract.py --load_json cifar10/vgg16/lid/k20/bim_8255.json
python extract.py --load_json cifar10/vgg16/lid/k20/pgd_8255.json
python extract.py --load_json cifar10/vgg16/lid/k20/aa_8255.json
python extract.py --load_json cifar10/vgg16/lid/k20/df.json
python extract.py --load_json cifar10/vgg16/lid/k20/cw.json
