class LearnableSerieslinearUnit(nn.ReLU):
    def __init__(self, dim, num_activations=3, deploy=False):
        super(LearnableSerieslinearUnit, self).__init__()
        self.num_activations = num_activations
        self.dim = dim
        self.deploy = deploy
        self.weight = torch.nn.Parameter(
            torch.randn(self.dim, 1, self.num_activations * 2 + 1, self.num_activations * 2 + 1))

        self.alphas = nn.Parameter(torch.ones(self.dim))
        self.biases = nn.Parameter(torch.zeros(self.dim))

        if self.deploy:
            self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(self.dim, eps=1e-6)

        self.activations = nn.ModuleList([nn.ReLU() for _ in range(self.num_activations)])

        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        if self.deploy:
            return F.conv2d(super(LearnableSerieslinearUnit, self).forward(x),
                            self.weight * self.alphas.view(-1, 1, 1, 1), self.bias, padding=self.num_activations,
                            groups=self.dim)
        else:
            for i in range(self.num_activations):
                x = self.activations[i](x)
                x = x * self.alphas[i] + self.biases[i]

            return self.bn(F.conv2d(
                    super(LearnableSerieslinearUnit, self).forward(x),
                    self.weight, None, padding=self.num_activations, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True
