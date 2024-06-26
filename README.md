# Learnable series linear units (LSLU)
## 1.Introduction
The effective activation functions enhance the fitting capability of neural networks by introducing various forms of non-linear transformations, enabling them to better adapt to real data distributions. We propose a series-based learnable activation function, LSLU (Learnable Series Linear Units), which renders various deep learning networks more concise and accurate. We introduce learnable parameters θ and ω to control the amplitude and slope of the activation function's oscillation. This approach increases the non-linearity of each activation function layer, thereby enhancing the overall network non-linearity and reducing the depth of neural networks. We evaluate the performance of LSLU on the CIFAR-10 and CIFAR-100 datasets and validate its effectiveness on a specific task dataset (Silkworm).
## 2.Code
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
We introduce learnable parameters  θ and  ω to control the oscillation amplitude and slope of the activation function, given by

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/b4c63330-5edd-404d-8b30-fcc4db57601f)

The parameters θ  and ω , initialized to 1 and 0, respectively, are updated based on backpropagation during training.
![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/7d9ba6f4-6655-46b5-9a9d-810e0c2a1c65)

the weight matrix is further adjusted based on the parameter θ .

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/73183f03-7460-4273-adc0-43d63a69433a)

Changes in LSLU：

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/65517ba5-9eb8-4b15-b4a5-6605728e5a21)

The detailed code can be found in `LSLU.py`.
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

output `learnable_params.txt`:

```
if args.output_dir and utils.is_main_process():
            output_dir = args.output_dir
            output_file_path = os.path.join(output_dir, f"learnable_params.txt")  
            with open(output_file_path, "a") as f:
                f.write(f'Epoch {epoch + 1}:\n')
                for name, param in model.named_parameters():
                    if 'alphas' in name or 'biases' in name:
                        values = param.tolist()
                        f.write(f'{name}: {values}\n')
                        print(f'learnable_params_{name}_out\n')
                f.write('\n')
```

## 3.Experiments
`CIFAR-10` Classification Results:

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/5d036d3a-3ed3-4b5b-8b9d-b12d4f2871e4)


`CIFAR-100` Classification Results: experimental results of ResNet-LSLU and VanillaNet-LSLU on CIFAR-100 when the number of activation functions is varied.

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/a16afc18-c42b-4823-bf8b-fbe5ef83f18c)

Experimental results of ResNet18-LSLU and VanillaNet5-LSLU on CIFAR-100 when the number of activation functions is varied. When training neural networks, the form of the activation function (slope and oscillation coefficient) should not remain constant.

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/8c6d5cb5-cade-4d12-8f26-4f68ca828ca3)

The impact of `regularization` strength.

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/2931310f-24aa-4bee-a019-672a14816a41)


`Silkworm` Classification Results:

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/fa51b029-f27f-44cb-bd17-9ef02f448198)



## 4.Modification Suggestions
we recommend using LSLU judiciously based on information such as the network's structure and depth: for shallow networks like VanillaNet, LSLU can be fully substituted for the old activation functions. However, for deep networks such as ResNet, and EfficientNetV2, which inherently possess high nonlinearity, careful consideration is required when selecting the position, number of activation functions, and the values of dropout rates for LSLU usage.

## 5.Acknowledgement
This repository is built using the timm library, DeiT, BEiT, RegVGG, ConvNeXt and VanillaNet repositories.

## 6.installation
The results are produced with `torch==1.10.2+cu113` `torchvision==0.11.3+cu113` `timm==0.6.12`. Other versions might also work.
Install Pytorch and, torchvision following official instructions.

Install required packages:
```
pip install timm==0.6.12
pip install cupy-cuda113
pip install torchprofile
pip install einops
pip install tensorboardX
pip install terminaltables
```
