# # Generate data
python gen.py --load_json cifar10/wrn28-10/fgsm_8255.json 
python gen.py --load_json cifar10/wrn28-10/bim_8255.json  
python gen.py --load_json cifar10/wrn28-10/pgd_8255.json  
python gen.py --load_json cifar10/wrn28-10/aa_8255.json   
python gen.py --load_json cifar10/wrn28-10/cw.json        
python gen.py --load_json cifar10/wrn28-10/df.json        

python gen.py --load_json cifar10/vgg16/fgsm_8255.json     
python gen.py --load_json cifar10/vgg16/bim_8255.json     
python gen.py --load_json cifar10/vgg16/pgd_8255.json      
python gen.py --load_json cifar10/vgg16/aa_8255.json       
python gen.py --load_json cifar10/vgg16/df.json               
python gen.py --load_json cifar10/vgg16/cw.json              

python gen.py --load_json cifar100/wrn28-10/fgsm_8255.json 
python gen.py --load_json cifar100/wrn28-10/bim_8255.json   
python gen.py --load_json cifar100/wrn28-10/pgd_8255.json    
python gen.py --load_json cifar100/wrn28-10/aa_8255.json    
python gen.py --load_json cifar100/wrn28-10/cw.json         
python gen.py --load_json cifar100/wrn28-10/df.json         

python gen.py --load_json cifar100/vgg16/fgsm_8255.json    
python gen.py --load_json cifar100/vgg16/bim_8255.json     
python gen.py --load_json cifar100/vgg16/pgd_8255.json        
python gen.py --load_json cifar100/vgg16/aa_8255.json       
python gen.py --load_json cifar100/vgg16/cw.json                 
python gen.py --load_json cifar100/vgg16/df.json                       

python gen.py --load_json imagenet/wrn50-2/fgsm_8255.json 
python gen.py --load_json imagenet/wrn50-2/bim_8255.json   
python gen.py --load_json imagenet/wrn50-2/pgd_8255.json  
python gen.py --load_json imagenet/wrn50-2/aa_8255.json   
python gen.py --load_json imagenet/wrn50-2/df.json       
python gen.py --load_json imagenet/wrn50-2/cw.json        


# # run 2
python gen.py --load_json cifar10/wrn28-10/fgsm_8255.json --run_nr run_2 --shuffle True
python gen.py --load_json cifar10/wrn28-10/bim_8255.json  --run_nr run_2 --shuffle True
python gen.py --load_json cifar10/wrn28-10/pgd_8255.json  --run_nr run_2 --shuffle True
python gen.py --load_json cifar10/wrn28-10/aa_8255.json   --run_nr run_2 --shuffle True
python gen.py --load_json cifar10/wrn28-10/cw.json        --run_nr run_2 --shuffle True
python gen.py --load_json cifar10/wrn28-10/df.json        --run_nr run_2 --shuffle True

python gen.py --load_json cifar10/vgg16/fgsm_8255.json    --run_nr run_2 --shuffle True 
python gen.py --load_json cifar10/vgg16/bim_8255.json     --run_nr run_2 --shuffle True
python gen.py --load_json cifar10/vgg16/pgd_8255.json     --run_nr run_2 --shuffle True 
python gen.py --load_json cifar10/vgg16/aa_8255.json      --run_nr run_2 --shuffle True 
python gen.py --load_json cifar10/vgg16/df.json           --run_nr run_2 --shuffle True    
python gen.py --load_json cifar10/vgg16/cw.json           --run_nr run_2 --shuffle True   

python gen.py --load_json cifar100/wrn28-10/fgsm_8255.json --run_nr run_2 --shuffle True
python gen.py --load_json cifar100/wrn28-10/bim_8255.json  --run_nr run_2 --shuffle True 
python gen.py --load_json cifar100/wrn28-10/pgd_8255.json  --run_nr run_2 --shuffle True  
python gen.py --load_json cifar100/wrn28-10/aa_8255.json   --run_nr run_2 --shuffle True 
python gen.py --load_json cifar100/wrn28-10/cw.json        --run_nr run_2 --shuffle True 
python gen.py --load_json cifar100/wrn28-10/df.json        --run_nr run_2 --shuffle True 

python gen.py --load_json cifar100/vgg16/fgsm_8255.json    --run_nr run_2 --shuffle True
python gen.py --load_json cifar100/vgg16/bim_8255.json     --run_nr run_2 --shuffle True
python gen.py --load_json cifar100/vgg16/pgd_8255.json     --run_nr run_2 --shuffle True   
python gen.py --load_json cifar100/vgg16/aa_8255.json      --run_nr run_2 --shuffle True 
python gen.py --load_json cifar100/vgg16/cw.json           --run_nr run_2 --shuffle True      
python gen.py --load_json cifar100/vgg16/df.json           --run_nr run_2 --shuffle True            


python gen.py --load_json imagenet/wrn50-2/fgsm_8255.json --run_nr run_2  
python gen.py --load_json imagenet/wrn50-2/bim_8255.json  --run_nr run_2  
python gen.py --load_json imagenet/wrn50-2/pgd_8255.json  --run_nr run_2  
python gen.py --load_json imagenet/wrn50-2/aa_8255.json   --run_nr run_2  
python gen.py --load_json imagenet/wrn50-2/df.json        --run_nr run_2  
python gen.py --load_json imagenet/wrn50-2/cw.json        --run_nr run_2  


# # run 3
python gen.py --load_json cifar10/wrn28-10/fgsm_8255.json --run_nr run_3 --shuffle True
python gen.py --load_json cifar10/wrn28-10/bim_8255.json  --run_nr run_3 --shuffle True
python gen.py --load_json cifar10/wrn28-10/pgd_8255.json  --run_nr run_3 --shuffle True
python gen.py --load_json cifar10/wrn28-10/aa_8255.json   --run_nr run_3 --shuffle True
python gen.py --load_json cifar10/wrn28-10/cw.json        --run_nr run_3 --shuffle True
python gen.py --load_json cifar10/wrn28-10/df.json        --run_nr run_3 --shuffle True

python gen.py --load_json cifar10/vgg16/fgsm_8255.json    --run_nr run_3 --shuffle True 
python gen.py --load_json cifar10/vgg16/bim_8255.json     --run_nr run_3 --shuffle True
python gen.py --load_json cifar10/vgg16/pgd_8255.json     --run_nr run_3 --shuffle True 
python gen.py --load_json cifar10/vgg16/aa_8255.json      --run_nr run_3 --shuffle True 
python gen.py --load_json cifar10/vgg16/df.json           --run_nr run_3 --shuffle True    
python gen.py --load_json cifar10/vgg16/cw.json           --run_nr run_3 --shuffle True   

python gen.py --load_json cifar100/wrn28-10/fgsm_8255.json --run_nr run_3 --shuffle True
python gen.py --load_json cifar100/wrn28-10/bim_8255.json  --run_nr run_3 --shuffle True 
python gen.py --load_json cifar100/wrn28-10/pgd_8255.json  --run_nr run_3 --shuffle True  
python gen.py --load_json cifar100/wrn28-10/aa_8255.json   --run_nr run_3 --shuffle True 
python gen.py --load_json cifar100/wrn28-10/cw.json        --run_nr run_3 --shuffle True 
python gen.py --load_json cifar100/wrn28-10/df.json        --run_nr run_3 --shuffle True 

python gen.py --load_json cifar100/vgg16/fgsm_8255.json    --run_nr run_3 --shuffle True
python gen.py --load_json cifar100/vgg16/bim_8255.json     --run_nr run_3 --shuffle True
python gen.py --load_json cifar100/vgg16/pgd_8255.json     --run_nr run_3 --shuffle True   
python gen.py --load_json cifar100/vgg16/aa_8255.json      --run_nr run_3 --shuffle True 
python gen.py --load_json cifar100/vgg16/cw.json           --run_nr run_3 --shuffle True      
python gen.py --load_json cifar100/vgg16/df.json           --run_nr run_3 --shuffle True            

python gen.py --load_json imagenet/wrn50-2/fgsm_8255.json --run_nr run_3 
python gen.py --load_json imagenet/wrn50-2/bim_8255.json  --run_nr run_3 
python gen.py --load_json imagenet/wrn50-2/pgd_8255.json  --run_nr run_3 
python gen.py --load_json imagenet/wrn50-2/aa_8255.json   --run_nr run_3 
python gen.py --load_json imagenet/wrn50-2/df.json        --run_nr run_3 
python gen.py --load_json imagenet/wrn50-2/cw.json        --run_nr run_3 





# Extract Data

python extract.py --load_json cifar10/vgg16/multilid/k20/fgsm_8255.json
python extract.py --load_json cifar10/vgg16/multilid/k20/bim_8255.json
python extract.py --load_json cifar10/vgg16/multilid/k20/pgd_8255.json
python extract.py --load_json cifar10/vgg16/multilid/k20/aa_8255.json
python extract.py --load_json cifar10/vgg16/multilid/k20/df.json
python extract.py --load_json cifar10/vgg16/multilid/k20/cw.json

python extract.py --load_json cifar10/wrn28-10/multilid/k20/fgsm_8255.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/bim_8255.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/pgd_8255.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/aa_8255.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/df.json
python extract.py --load_json cifar10/wrn28-10/multilid/k20/cw.json

python extract.py --load_json cifar10/vgg16/lid/k20/fgsm_8255.json
python extract.py --load_json cifar10/vgg16/lid/k20/bim_8255.json
python extract.py --load_json cifar10/vgg16/lid/k20/pgd_8255.json
python extract.py --load_json cifar10/vgg16/lid/k20/aa_8255.json
python extract.py --load_json cifar10/vgg16/lid/k20/df.json
python extract.py --load_json cifar10/vgg16/lid/k20/cw.json

python extract.py --load_json cifar10/wrn28-10/lid/k20/fgsm_8255.json
python extract.py --load_json cifar10/wrn28-10/lid/k20/bim_8255.json
python extract.py --load_json cifar10/wrn28-10/lid/k20/pgd_8255.json
python extract.py --load_json cifar10/wrn28-10/lid/k20/aa_8255.json
python extract.py --load_json cifar10/wrn28-10/lid/k20/df.json
python extract.py --load_json cifar10/wrn28-10/lid/k20/cw.json

python extract.py --load_json cifar100/vgg16/multilid/k20/fgsm_8255.json
python extract.py --load_json cifar100/vgg16/multilid/k20/bim_8255.json
python extract.py --load_json cifar100/vgg16/multilid/k20/pgd_8255.json
python extract.py --load_json cifar100/vgg16/multilid/k20/aa_8255.json
python extract.py --load_json cifar100/vgg16/multilid/k20/df.json
python extract.py --load_json cifar100/vgg16/multilid/k20/cw.json

python extract.py --load_json cifar100/wrn28-10/multilid/k20/fgsm_8255.json
python extract.py --load_json cifar100/wrn28-10/multilid/k20/bim_8255.json
python extract.py --load_json cifar100/wrn28-10/multilid/k20/pgd_8255.json
python extract.py --load_json cifar100/wrn28-10/multilid/k20/aa_8255.json
python extract.py --load_json cifar100/wrn28-10/multilid/k20/df.json
python extract.py --load_json cifar100/wrn28-10/multilid/k20/cw.json

python extract.py --load_json cifar100/vgg16/lid/k20/fgsm_8255.json
python extract.py --load_json cifar100/vgg16/lid/k20/bim_8255.json
python extract.py --load_json cifar100/vgg16/lid/k20/pgd_8255.json
python extract.py --load_json cifar100/vgg16/lid/k20/aa_8255.json
python extract.py --load_json cifar100/vgg16/lid/k20/df.json
python extract.py --load_json cifar100/vgg16/lid/k20/cw.json

python extract.py --load_json cifar100/wrn28-10/lid/k20/fgsm_8255.json
python extract.py --load_json cifar100/wrn28-10/lid/k20/bim_8255.json
python extract.py --load_json cifar100/wrn28-10/lid/k20/pgd_8255.json
python extract.py --load_json cifar100/wrn28-10/lid/k20/aa_8255.json
python extract.py --load_json cifar100/wrn28-10/lid/k20/df.json
python extract.py --load_json cifar100/wrn28-10/lid/k20/cw.json

python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_8255.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/bim_8255.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/pgd_8255.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/aa_8255.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/df.json
python extract.py --load_json imagenet/wrn50-2/multilid/k30/cw.json

python extract.py --load_json imagenet/wrn50-2/lid/k30/fgsm_8255.json
python extract.py --load_json imagenet/wrn50-2/lid/k30/bim_8255.json
python extract.py --load_json imagenet/wrn50-2/lid/k30/pgd_8255.json
python extract.py --load_json imagenet/wrn50-2/lid/k30/aa_8255.json
python extract.py --load_json imagenet/wrn50-2/lid/k30/df.json
python extract.py --load_json imagenet/wrn50-2/lid/k30/cw.json


python extract.py --load_json cifar10/vgg16/multilid/k20/fgsm_8255.json --run_nr run_2 
python extract.py --load_json cifar10/vgg16/multilid/k20/bim_8255.json  --run_nr run_2  
python extract.py --load_json cifar10/vgg16/multilid/k20/pgd_8255.json  --run_nr run_2    
python extract.py --load_json cifar10/vgg16/multilid/k20/aa_8255.json   --run_nr run_2  
python extract.py --load_json cifar10/vgg16/multilid/k20/df.json        --run_nr run_2      
python extract.py --load_json cifar10/vgg16/multilid/k20/cw.json        --run_nr run_2    

python extract.py --load_json cifar10/wrn28-10/multilid/k20/fgsm_8255.json --run_nr run_2   
python extract.py --load_json cifar10/wrn28-10/multilid/k20/bim_8255.json  --run_nr run_2 
python extract.py --load_json cifar10/wrn28-10/multilid/k20/pgd_8255.json  --run_nr run_2 
python extract.py --load_json cifar10/wrn28-10/multilid/k20/aa_8255.json   --run_nr run_2    
python extract.py --load_json cifar10/wrn28-10/multilid/k20/df.json        --run_nr run_2      
python extract.py --load_json cifar10/wrn28-10/multilid/k20/cw.json        --run_nr run_2     

python extract.py --load_json cifar10/vgg16/lid/k20/fgsm_8255.json --run_nr run_2      
python extract.py --load_json cifar10/vgg16/lid/k20/bim_8255.json  --run_nr run_2      
python extract.py --load_json cifar10/vgg16/lid/k20/pgd_8255.json  --run_nr run_2       
python extract.py --load_json cifar10/vgg16/lid/k20/aa_8255.json   --run_nr run_2           
python extract.py --load_json cifar10/vgg16/lid/k20/df.json        --run_nr run_2      
python extract.py --load_json cifar10/vgg16/lid/k20/cw.json        --run_nr run_2      

python extract.py --load_json cifar10/wrn28-10/lid/k20/fgsm_8255.jso --run_nr run_2 
python extract.py --load_json cifar10/wrn28-10/lid/k20/bim_8255.json --run_nr run_2  
python extract.py --load_json cifar10/wrn28-10/lid/k20/pgd_8255.json --run_nr run_2   
python extract.py --load_json cifar10/wrn28-10/lid/k20/aa_8255.json  --run_nr run_2 
python extract.py --load_json cifar10/wrn28-10/lid/k20/df.json       --run_nr run_2    
python extract.py --load_json cifar10/wrn28-10/lid/k20/cw.json       --run_nr run_2       

python extract.py --load_json cifar100/vgg16/multilid/k20/fgsm_8255.json --run_nr run_2 
python extract.py --load_json cifar100/vgg16/multilid/k20/bim_8255.json  --run_nr run_2  
python extract.py --load_json cifar100/vgg16/multilid/k20/pgd_8255.json  --run_nr run_2   
python extract.py --load_json cifar100/vgg16/multilid/k20/aa_8255.json   --run_nr run_2  
python extract.py --load_json cifar100/vgg16/multilid/k20/df.json        --run_nr run_2   
python extract.py --load_json cifar100/vgg16/multilid/k20/cw.json        --run_nr run_2       

python extract.py --load_json cifar100/wrn28-10/multilid/k20/fgsm_8255.json --run_nr run_2  
python extract.py --load_json cifar100/wrn28-10/multilid/k20/bim_8255.json  --run_nr run_2   
python extract.py --load_json cifar100/wrn28-10/multilid/k20/pgd_8255.json  --run_nr run_2      
python extract.py --load_json cifar100/wrn28-10/multilid/k20/aa_8255.json   --run_nr run_2     
python extract.py --load_json cifar100/wrn28-10/multilid/k20/df.json        --run_nr run_2     
python extract.py --load_json cifar100/wrn28-10/multilid/k20/cw.json        --run_nr run_2     

python extract.py --load_json cifar100/vgg16/lid/k20/fgsm_8255.json     --run_nr run_2     
python extract.py --load_json cifar100/vgg16/lid/k20/bim_8255.json      --run_nr run_2    
python extract.py --load_json cifar100/vgg16/lid/k20/pgd_8255.json      --run_nr run_2     
python extract.py --load_json cifar100/vgg16/lid/k20/aa_8255.json       --run_nr run_2      
python extract.py --load_json cifar100/vgg16/lid/k20/df.json            --run_nr run_2    
python extract.py --load_json cifar100/vgg16/lid/k20/cw.json            --run_nr run_2  

python extract.py --load_json cifar100/wrn28-10/lid/k20/fgsm_8255.json --run_nr run_2    
python extract.py --load_json cifar100/wrn28-10/lid/k20/bim_8255.json  --run_nr run_2       
python extract.py --load_json cifar100/wrn28-10/lid/k20/pgd_8255.json  --run_nr run_2      
python extract.py --load_json cifar100/wrn28-10/lid/k20/aa_8255.json   --run_nr run_2     
python extract.py --load_json cifar100/wrn28-10/lid/k20/df.json        --run_nr run_2    
python extract.py --load_json cifar100/wrn28-10/lid/k20/cw.json        --run_nr run_2 

python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_8255.json --run_nr run_2   
python extract.py --load_json imagenet/wrn50-2/multilid/k30/bim_8255.json  --run_nr run_2      
python extract.py --load_json imagenet/wrn50-2/multilid/k30/pgd_8255.json  --run_nr run_2    
python extract.py --load_json imagenet/wrn50-2/multilid/k30/aa_8255.json   --run_nr run_2       
python extract.py --load_json imagenet/wrn50-2/multilid/k30/df.json        --run_nr run_2     
python extract.py --load_json imagenet/wrn50-2/multilid/k30/cw.json        --run_nr run_2      

python extract.py --load_json imagenet/wrn50-2/lid/k30/fgsm_8255.json --run_nr run_2     
python extract.py --load_json imagenet/wrn50-2/lid/k30/bim_8255.json  --run_nr run_2    
python extract.py --load_json imagenet/wrn50-2/lid/k30/pgd_8255.json  --run_nr run_2     
python extract.py --load_json imagenet/wrn50-2/lid/k30/aa_8255.json   --run_nr run_2     
python extract.py --load_json imagenet/wrn50-2/lid/k30/df.json        --run_nr run_2            
python extract.py --load_json imagenet/wrn50-2/lid/k30/cw.json        --run_nr run_2      


python extract.py --load_json cifar10/vgg16/multilid/k20/fgsm_8255.json --run_nr run_3 
python extract.py --load_json cifar10/vgg16/multilid/k20/bim_8255.json  --run_nr run_3  
python extract.py --load_json cifar10/vgg16/multilid/k20/pgd_8255.json  --run_nr run_3    
python extract.py --load_json cifar10/vgg16/multilid/k20/aa_8255.json   --run_nr run_3  
python extract.py --load_json cifar10/vgg16/multilid/k20/df.json        --run_nr run_3      
python extract.py --load_json cifar10/vgg16/multilid/k20/cw.json        --run_nr run_3    

python extract.py --load_json cifar10/wrn28-10/multilid/k20/fgsm_8255.json --run_nr run_3   
python extract.py --load_json cifar10/wrn28-10/multilid/k20/bim_8255.json  --run_nr run_3 
python extract.py --load_json cifar10/wrn28-10/multilid/k20/pgd_8255.json  --run_nr run_3 
python extract.py --load_json cifar10/wrn28-10/multilid/k20/aa_8255.json   --run_nr run_3    
python extract.py --load_json cifar10/wrn28-10/multilid/k20/df.json        --run_nr run_3      
python extract.py --load_json cifar10/wrn28-10/multilid/k20/cw.json        --run_nr run_3     

python extract.py --load_json cifar10/vgg16/lid/k20/fgsm_8255.json --run_nr run_3      
python extract.py --load_json cifar10/vgg16/lid/k20/bim_8255.json  --run_nr run_3      
python extract.py --load_json cifar10/vgg16/lid/k20/pgd_8255.json  --run_nr run_3       
python extract.py --load_json cifar10/vgg16/lid/k20/aa_8255.json   --run_nr run_3           
python extract.py --load_json cifar10/vgg16/lid/k20/df.json        --run_nr run_3      
python extract.py --load_json cifar10/vgg16/lid/k20/cw.json        --run_nr run_3      

python extract.py --load_json cifar10/wrn28-10/lid/k20/fgsm_8255.jso --run_nr run_3 
python extract.py --load_json cifar10/wrn28-10/lid/k20/bim_8255.json --run_nr run_3  
python extract.py --load_json cifar10/wrn28-10/lid/k20/pgd_8255.json --run_nr run_3   
python extract.py --load_json cifar10/wrn28-10/lid/k20/aa_8255.json  --run_nr run_3 
python extract.py --load_json cifar10/wrn28-10/lid/k20/df.json       --run_nr run_3    
python extract.py --load_json cifar10/wrn28-10/lid/k20/cw.json       --run_nr run_3       

python extract.py --load_json cifar100/vgg16/multilid/k20/fgsm_8255.json --run_nr run_3 
python extract.py --load_json cifar100/vgg16/multilid/k20/bim_8255.json  --run_nr run_3  
python extract.py --load_json cifar100/vgg16/multilid/k20/pgd_8255.json  --run_nr run_3   
python extract.py --load_json cifar100/vgg16/multilid/k20/aa_8255.json   --run_nr run_3  
python extract.py --load_json cifar100/vgg16/multilid/k20/df.json        --run_nr run_3   
python extract.py --load_json cifar100/vgg16/multilid/k20/cw.json        --run_nr run_3       

python extract.py --load_json cifar100/wrn28-10/multilid/k20/fgsm_8255.json --run_nr run_3  
python extract.py --load_json cifar100/wrn28-10/multilid/k20/bim_8255.json  --run_nr run_3   
python extract.py --load_json cifar100/wrn28-10/multilid/k20/pgd_8255.json  --run_nr run_3      
python extract.py --load_json cifar100/wrn28-10/multilid/k20/aa_8255.json   --run_nr run_3     
python extract.py --load_json cifar100/wrn28-10/multilid/k20/df.json        --run_nr run_3     
python extract.py --load_json cifar100/wrn28-10/multilid/k20/cw.json        --run_nr run_3     

python extract.py --load_json cifar100/vgg16/lid/k20/fgsm_8255.json     --run_nr run_3     
python extract.py --load_json cifar100/vgg16/lid/k20/bim_8255.json      --run_nr run_3    
python extract.py --load_json cifar100/vgg16/lid/k20/pgd_8255.json      --run_nr run_3     
python extract.py --load_json cifar100/vgg16/lid/k20/aa_8255.json       --run_nr run_3      
python extract.py --load_json cifar100/vgg16/lid/k20/df.json            --run_nr run_3    
python extract.py --load_json cifar100/vgg16/lid/k20/cw.json            --run_nr run_3  

python extract.py --load_json cifar100/wrn28-10/lid/k20/fgsm_8255.json --run_nr run_3    
python extract.py --load_json cifar100/wrn28-10/lid/k20/bim_8255.json  --run_nr run_3       
python extract.py --load_json cifar100/wrn28-10/lid/k20/pgd_8255.json  --run_nr run_3      
python extract.py --load_json cifar100/wrn28-10/lid/k20/aa_8255.json   --run_nr run_3     
python extract.py --load_json cifar100/wrn28-10/lid/k20/df.json        --run_nr run_3    
python extract.py --load_json cifar100/wrn28-10/lid/k20/cw.json        --run_nr run_3 

python extract.py --load_json imagenet/wrn50-2/multilid/k30/fgsm_8255.json --run_nr run_3   
python extract.py --load_json imagenet/wrn50-2/multilid/k30/bim_8255.json  --run_nr run_3      
python extract.py --load_json imagenet/wrn50-2/multilid/k30/pgd_8255.json  --run_nr run_3    
python extract.py --load_json imagenet/wrn50-2/multilid/k30/aa_8255.json   --run_nr run_3       
python extract.py --load_json imagenet/wrn50-2/multilid/k30/df.json        --run_nr run_3     
python extract.py --load_json imagenet/wrn50-2/multilid/k30/cw.json        --run_nr run_3      

python extract.py --load_json imagenet/wrn50-2/lid/k30/fgsm_8255.json --run_nr run_3     
python extract.py --load_json imagenet/wrn50-2/lid/k30/bim_8255.json  --run_nr run_3    
python extract.py --load_json imagenet/wrn50-2/lid/k30/pgd_8255.json  --run_nr run_3     
python extract.py --load_json imagenet/wrn50-2/lid/k30/aa_8255.json   --run_nr run_3     
python extract.py --load_json imagenet/wrn50-2/lid/k30/df.json        --run_nr run_3            
python extract.py --load_json imagenet/wrn50-2/lid/k30/cw.json        --run_nr run_3   