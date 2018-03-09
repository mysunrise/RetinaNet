from context import *


def test():
    input = Variable(torch.randn(1, 3, 600, 600))
    net = mymodels.fpn50(pretrained=True)
    resnet = torchvision.models.resnet50(pretrained=True)
    #print(net)
    if torch.cuda.is_available():
        input, net, resnet = input.cuda(), net.cuda(), resnet.cuda()
    output = net(input)
    for fmp in output:
        print(fmp.size())

    net_s_d = net.state_dict()
    resnet_s_d = resnet.state_dict()
    print(net_s_d.keys())
    print(resnet_s_d.keys())
    '''
    for key in net_s_d.keys():
        if key in resnet_s_d.keys():
            print(key)
            print(net_s_d[key] - resnet_s_d[key])
    
    for name, param in net.named_parameters():
        print(name, param.data)
    '''

if __name__ == '__main__':
    test()
