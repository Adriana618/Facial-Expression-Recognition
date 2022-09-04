from tools import *
import cv2

def showimg(data):
    img = data
    plt.figure()
    plt.imshow(torch.squeeze(img), cmap='gray')
    plt.show()

def run():
    train_loader, valid_loader, *rest = get_FEDdataset(1)
    for batch_num, batch in enumerate(train_loader, 7):
        a,p,n,l,pl,nl = batch
        print(l)
        print(pl)
        print(nl)
        print("------------")
        showimg(a)
        showimg(p)
        showimg(n)
        
if __name__ == '__main__':
    
    run()