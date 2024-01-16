# Learnable series linear units (LSLU)
## Introduction
The effective activation functions enhance the fitting capability of neural networks by introducing various forms of non-linear transformations, enabling them to better adapt to real data distributions. We propose a series-based learnable activation function, LSLU (Learnable Series Linear Units), which renders various deep learning networks more concise and accurate. We introduce learnable parameters α and β to control the amplitude and slope of the activation function's oscillation. This approach increases the non-linearity of each activation function layer, thereby enhancing the overall network non-linearity and reducing the depth of neural networks. We evaluate the performance of LSLU on the CIFAR-10 and CIFAR-100 datasets and validate its effectiveness on a specific task dataset (Silkworm).
## Code
Main code for LSLU:
```
# Creating multiple activation functions' coefficients (α) and biases (b) as learnable parameters
self.alphas = nn.Parameter(torch.ones(self.dim))
self.biases = nn.Parameter(torch.zeros(self.dim))       
# Creating multiple activation functions and placing them into a ModuleList
self.activations = nn.ModuleList([nn.ReLU() for _ in range(self.num_activations)])
if self.deploy:
            # In deployment mode, utilize convolutional operations.
            return F.conv2d(super(LearnableActivationWithBN, self).forward(x),
                            self.weight * self.alphas.view(-1, 1, 1, 1), self.bias, padding=self.num_activations,
                            groups=self.dim)
else:
            # In training mode, sequentially apply multiple activation functions and scale and bias them based on α and b.
            for i in range(self.num_activations):
                x = self.activations[i](x)  # Utilize the activation functions in the ModuleList.
                x = x * self.alphas[i] + self.biases[i]

            return self.bn(torch.nn.functional.conv2d(
                super(LearnableActivationWithBN, self).forward(x),
                self.weight, None, padding=self.num_activations, groups=self.dim))        
```
![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/65517ba5-9eb8-4b15-b4a5-6605728e5a21)

The usage `example` of LSLU:
```
self.lslu = LearnableSerieslinearUnit(self.in_channel, act_num, deploy=False)
```
Early stopping mechanism related code:

```
if test_loss is not None:
            score = -test_loss
            if best_score is None:
                best_score = score
            elif score < best_score + delta:
                counter += 1
                print(f'EarlyStopping counter: {counter} out of {args.early_stop_epochs}')
                if counter >= args.early_stop_epochsargs:
                    early_stop = 1
            else:
                best_score = score
                counter = 0

        if args.early_stop_epochs and early_stop == 1:
            break
```
