import time
import torch
from VanillaNet_LSLU import vanillanet_5 as create_model

if __name__ == "__main__":
    from timm.data import create_dataset, create_loader

    dataset_val = create_dataset(name='', root='./CIFAR10', split='validation', is_training=False, batch_size=1)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    size = 224
    data_loader_val = create_loader(dataset_val, input_size=size, batch_size=1, is_training=False, use_prefetcher=False)

    net = create_model(deploy=True).cuda()
    net.eval()
    print(net)
    for img, target in data_loader_val:
        img = img.cuda()
        for i in range(5):
            net(img)
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad():
            for i in range(1000):
                net(img)
                torch.cuda.synchronize()
        print((time.time() - t))  # The time required for 1000 inference runs.

        n_parameters = sum(p.numel() for p in net.parameters())
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        from torchprofile import profile_macs

        macs = profile_macs(net, img)
        print('model flops (G):', macs / 1.e9, 'input_size:', img.shape)

        break
