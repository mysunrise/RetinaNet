from context import *


def test():
    input = Variable(torch.randn(1, 3, 300, 300))
    net = mymodels.retinanet50(pretrained=True)
    resnet = torchvision.models.resnet50(pretrained=True)
    #print(net)
    for module in net.classifier.children():
        print module
    print((list(net.classifier.children())[-2]).bias.data)
    '''
    if torch.cuda.is_available():
        input, net, resnet = input.cuda(), net.cuda(), resnet.cuda()
    '''
    pred_cls, pred_loc = net(input)
    #print(pred_cls.size())
    #print(pred_loc.size())
    '''
    for name, param in net.named_parameters():
        print(name)
    print(net.state_dict()['backbone.conv1.weight'] - resnet.state_dict()['conv1.weight'])
    '''

if __name__ == '__main__':
    test()
