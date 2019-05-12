r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 1
    lr = 0.03
    reg = 0.1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 10
    lr_vanilla = 0.03
    lr_momentum = 0.005
    lr_rmsprop = 0.0007
    reg = 0.001
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 1
    lr = 0.005
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**
1. The no-dropout train accuracy graph rapidly gets to high accuracy values, and low loss values.
In the no-dropout test accuracy graph we see low accuracy values and high loss. As expected the network
quickly adapted to the small training set, but learned very little for the general case.

The no-dropout train accuracy graph rapidly gets to high accuracy values, and low loss values.
in contrary to the dropout train accuracy graphs which improves their accuracy and loss on a slower rate.
That phenomena is expected because when passing forward over the training set,
we choose different neurons to train in every batch, therefor a neuron trained on a previous batch
might no be included on this batch's training and will not have any effect on the classifier.
That also explains the training loss graphs. the no-dropout graph quickly gets to low loss values,
while the dropout graphs' descent rate is slower.

In the no-dropout test loss graph we see a rapid increase in loss values.
The dropout test loss graphs are low, and even descending on a very slow pace.
We expected that result as well because having dropout while training allows the network to generalize
much better and therefor maintain low loss values.

The test accuracy graphs have similar values through the tests, despite the no-dropout's higher loss values.
We think that the problem is the size of the training set, which is very small. The no-dropout network has higher loss
because of the over-fitting to the training set, that is the reason that the accuracy is low.
The dropout networks present less over-fitting (or even not at all), therefor lower loss, but still low accuracy
because of low train set accuracy as well. For dropout networks, train and test accuracies are more coherent.

2. From the graphs we can see that low dropout value leads to the highest training loss but the lowest test loss.
We also can see from the graphs that the test and train accuracies for low dropout are similar, which means that 
what is being learned, is also being generalized.
Higher dropout value leads to higher over-fit, but not as high as no dropout (We can see higher train accuracies then
test accuracies, which implies for some over-fit).
**

"""

part2_q2 = r"""
**
It is possible for the test loss to increase for few epochs while the accuracy is also increasing, when using
cross-entropy loss function.
Let us take a look at the loss formula: $- x_y + \log\left(\sum_k e^{x_k}\right)$
The first expression, $- x_y$, is the probability calculated by the neural network for sample x to be of class y
(the correct class). If this probability is the highest among the rest of the classes, we will count that as an
accurate calculation.
It means that as long as for sample x that belongs to class y, the appropriate index in the result vector is the highest
one, it will count as an accurate calculation.
If our probability vectors has extreme values, which means high value for the suspected correct class, and very low
values for the rest of the classes. During few epochs of training, we detect the correct class better
(more probability vectors' highest values indexes correspond to correct class), but the probability values across
a vector are more unified.
An example for an extreme probability vector (let's assume two classes classifier):
(0.9, 0.1) - sample loss = $-0.9 + log(e^{0.1})$
After a few epochs, where the extreme values are treated:
(p, 1-p) where p < 0.9 - sample loss = $-p + log(e^{1-p}) > -0.9 + log(e^{0.1})$**

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
