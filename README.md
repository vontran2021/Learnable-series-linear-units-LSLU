# Learnable series linear units (LSLU)
## Introduction
The effective activation functions enhance the fitting capability of neural networks by introducing various forms of non-linear transformations, enabling them to better adapt to real data distributions. We propose a series-based learnable activation function, LSLU (Learnable Series Linear Units), which renders various deep learning networks more concise and accurate. We introduce learnable parameters α and β to control the amplitude and slope of the activation function's oscillation. This approach increases the non-linearity of each activation function layer, thereby enhancing the overall network non-linearity and reducing the depth of neural networks. We evaluate the performance of LSLU on the CIFAR-10 and CIFAR-100 datasets and validate its effectiveness on a specific task dataset (Silkworm).
## Code
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
