from convnet.configuration_convnet import ConNetConfig
from convnet.modeling_convnet import ConvNetModel
import torch

import torchvision
import torchvision.transforms as transforms

if __name__=="__main__":
    model = ConvNetModel.from_pretrained("Zihua-Liu-CVer/custom-convnet")
    model.eval()
    batch_size =1
    '''Datasets'''
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)
    test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

    model.cuda()
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for i,(val_images,val_labels) in enumerate(test_loader):
            val_images = val_images.cuda()
            val_labels = val_labels.cuda()
            out = model(val_images)
            _, val_predicted = torch.max(out.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()
    
        val_acc_cur_epoch = 100 * val_correct/val_total
        print("accurate rate is {}".format(val_acc_cur_epoch))