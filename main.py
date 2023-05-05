import os
import time



if __name__ == '__main__':

    choice = input("Select options:\n1.TRY-ON\n2.TensorBoard\n")
    
    if int(choice) == 1:
        person = str(input("enter person pics separated by comma ")).split(',')
        cloth = str(input("enter cloth pic ")).split(',')
        
        if len(person)>0:
            with open('data//product_test.txt','w') as fp:
                for p in person:
                    for q in cloth:
                        final = p+'.jpg' + ' ' + q +'.jpg'
                        fp.write(final)
                        fp.write('\n')
        
        os.system('python test.py --name GMM --stage GMM  --checkpoint checkpoints/GMM/gmm_final.pth')
        time.sleep(2)
        os.system("python test.py --name TOM --stage TOM  --checkpoint checkpoints/TOM/tom_final.pth")
    elif int(choice) == 2:
        choice = input("launch tensorboard for:\n1.GMM\n2.TOM\n>> ")
        if int(choice) == 1:
            os.system('python -m tensorboard.main --logdir=tensorboard/GMM/')
        elif int(choice) == 2:
            os.system('python -m tensorboard.main --logdir=tensorboard/TOM/')
        else:
            print("Invalid")
    else:
        print("Invalid")