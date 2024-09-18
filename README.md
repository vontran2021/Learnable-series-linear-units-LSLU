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

![1721205649300](https://github.com/user-attachments/assets/bebcb6b9-4b66-4c5f-b822-01f886d0ce1c)

The parameters θ  and ω , initialized to 1 and 0, n denotes the number of activation functions. Respectively, are updated based on backpropagation during training.

![image](https://github.com/vontran2021/Learnable-series-linear-units-LSLU/assets/97432746/7d9ba6f4-6655-46b5-9a9d-810e0c2a1c65)

the weight matrix is further adjusted based on the parameter θ .

![image](https://github.com/user-attachments/assets/a4a06d92-e8b3-4d18-969b-4d2c240727d9)

Changes in LSLU：

![LSLU应用](https://github.com/user-attachments/assets/c39389d1-e3ab-429a-9055-ed8510ffa8ef)


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

![image](https://github.com/user-attachments/assets/5eb59e40-fd04-4ffc-9ace-3eb20f208f93)




## 4.Modification Suggestions
we recommend using LSLU judiciously based on information such as the network's structure and depth: for shallow networks like VanillaNet, LSLU can be fully substituted for the old activation functions. However, for deep networks such as ResNet, and EfficientNetV2, which inherently possess high nonlinearity, careful consideration is required when selecting the position, number of activation functions, and the values of dropout rates for LSLU usage.

## 5.Acknowledgement
This repository is built using the timm library, DeiT, BEiT, RegVGG, ConvNeXt and VanillaNet repositories.

## 6.Installation
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

## 7.testing
We give an example evaluation command for VanillaNet-5:
```
python -m torch.distributed.launch --nproc_per_node=1 main.py --model vanillanet_5 --data_path /path/to/dataset/ --finetune /path/to/vanillanet_5.pth --eval True --model_key model_ema --crop_pct 0.875
```
with deploy:
```
python -m torch.distributed.launch --nproc_per_node=1 main.py --model vanillanet_5 --data_path /path/to/dataset/ --finetune /path/to/vanillanet_5.pth --eval True --model_key model_ema --crop_pct 0.875 --switch_to_deploy /path/to/vanillanet_5_deploy.pth
```

## 8.Training
You can use the following command to train VanillaNet-5 on a single machine with 2 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model vanillanet_5 \
--data_path /path/to/dataset \
--batch_size 400 --update_freq 1  --epochs 300 --decay_epochs 100 \ 
--lr 3.5e-3 --weight_decay 0.35  --drop 0.05 \
--opt lamb --aa rand-m7-mstd0.5-inc1 --mixup 0.1 --bce_loss \
--output_dir /path/to/save_results \
--model_ema true --model_ema_eval true --model_ema_decay 0.99996 \
--use_amp true
```
Here, the effective batch size = `--nproc_per_node` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `2*400*1 = 800`.

## 9.Citation
If you find this useful in your research, please consider citing:
```
@article{feng2024lslu,
  title={Activation function optimization method: Learnable series linear units (LSLUs)},
  author={Chuan Feng, Xi Lin, Shiping Zhu, Hongkang Shi, Maojie Tang, Hua Huang},
  journal={arXiv preprint arXiv:2409.08283},
  year={2024}
}
```


