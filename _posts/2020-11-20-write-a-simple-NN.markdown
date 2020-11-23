---
toc: false
layout: post
description: Whats happening underneath the fastai package? (testing)
categories: [markdown, posts]
title: Writing a simple NN
permalink: /:title.html
published: false
---

## Baseline

- Get path to tensor
- Make stacked tensors
- Get mean from training data
- get distance (L1norm or L2 norm) of validation
- predict if 3 or 7
- Accuracy display

- plot spread from training data
- do for other numbers


```python
#hide
#!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```


```python
#hide
from fastai.vision.all import *
from fastbook import *

matplotlib.rc('image', cmap='Greys')
```


```python
#hide
path = untar_data(URLs.MNIST_SAMPLE)
```


```python
#hide
Path.BASE_PATH = path
```


```python
#hide
path.ls()
```




    (#3) [Path('labels.csv'),Path('valid'),Path('train')]




```python
#hide
threes_tr = (path/'train'/'3').ls().sorted()
sevens_tr = (path/'train'/'7').ls().sorted()
threes_vd = (path/'valid'/'3').ls().sorted()
sevens_vd = (path/'valid'/'7').ls().sorted()
threes_tr
```




    (#6131) [Path('train/3/10.png'),Path('train/3/10000.png'),Path('train/3/10011.png'),Path('train/3/10031.png'),Path('train/3/10034.png'),Path('train/3/10042.png'),Path('train/3/10052.png'),Path('train/3/1007.png'),Path('train/3/10074.png'),Path('train/3/10091.png')...]




```python
# stack em with list comprehension
def stack_tensors(paths):
    lcomp_tensors = [tensor(Image.open(o)) for o in paths]
    print (len(lcomp_tensors))
    return torch.stack(lcomp_tensors).float()/255
```


```python
stacked_threes_tr = stack_tensors(threes_tr)
stacked_threes_vd = stack_tensors(threes_vd)
stacked_sevens_tr = stack_tensors(sevens_tr)
stacked_sevens_vd = stack_tensors(sevens_vd)
```

    6131
    1010
    6265
    1028



```python
stacked_threes_tr.shape
```




    torch.Size([6131, 28, 28])




```python
show_image(stacked_threes_tr.mean((0)))
```




    <AxesSubplot:>




    
![png](/images/nn/output_10_1.png)
    



```python
def l1_norm(a,b): return (a-b).abs().mean((-1,-2))
# check what happens when you use l2_norm
```


```python
mean_3_2d = stacked_threes_tr.mean((0))
mean_7_2d = stacked_sevens_tr.mean((0))
show_image(mean_7_2d)
```




    <AxesSubplot:>




    
![png](output_12_1.png)
    



```python
def is_3(stacked_tensor,mean_3,mean_7): 
    return l1_norm(stacked_tensor, mean_3)<l1_norm(stacked_tensor,mean_7)
```


```python
accuracy_3s_1 = is_3(stacked_threes_vd,mean_3_2d,mean_7_2d).float().mean()
accuracy_3s_2 = 1-is_3(stacked_sevens_vd,mean_3_2d,mean_7_2d).float().mean()

print("Accuracy of prediction is: ",(accuracy_3s_1+accuracy_3s_2)/2)

# no need to do 7 separately
```

    Accuracy of prediction is:  tensor(0.9511)


## Write your own Linear NN


```python
gv('''
init->predict->loss->gradient->step->stop
step->predict[label=repeat]
''')
```




    
![svg](output_16_0.svg)
    



## Steps

- Initialize some parameters (w)
- need the pixel values. So let's reshape
- predict with those parameters (x*w).sum()
- find the loss with those parameters (distance norm)
- take the gradient based on the loss function
- make a step
- prediction of the validation set
- 20 times for loop


```python
## Concatenate the 7s and 3s
train_x=torch.cat((stacked_threes_tr,stacked_sevens_tr),0).view(-1,28*28)
train_y=tensor(([1]*len(stacked_threes_tr)+[0]*len(stacked_sevens_tr))).unsqueeze(1)
valid_x=torch.cat((stacked_threes_vd,stacked_sevens_vd),0).view(-1,28*28)
valid_y=tensor(([1]*len(stacked_threes_vd)+[0]*len(stacked_sevens_vd))).unsqueeze(1)

train_x.shape,stacked_threes_tr.shape,stacked_sevens_tr.shape,train_y.shape,valid_x.shape,stacked_threes_vd.shape,stacked_sevens_vd.shape
```




    (torch.Size([12396, 784]),
     torch.Size([6131, 28, 28]),
     torch.Size([6265, 28, 28]),
     torch.Size([12396, 1]),
     torch.Size([2038, 784]),
     torch.Size([1010, 28, 28]),
     torch.Size([1028, 28, 28]))




```python
## Squeeze into tuple
dset= list(zip(train_x,train_y))
valid_dset = list(zip(valid_x,valid_y))
x,y = dset[0]

x.shape,y,len(dset),type(dset[0]),dset[12000][1]
```




    (torch.Size([784]), tensor([1]), 12396, tuple, tensor([0]))




```python
train_y.shape
```




    torch.Size([12396, 1])




```python
# z = torch.arange(0,len(train_y),1).unsqueeze(1)
# z.shape,z,train_y.shape
```


```python
# # Squeeze into tuple
# dset= list(zip(train_x,train_y,z))
# x,y,z = dset[0]

# x.shape,y,z,len(dset),type(dset[0]),dset[12000][1]
```


```python
# dl = DataLoader(dset,batch_size=20,shuffle=True)
# xb,yb,zb = first(dl)
# xb.shape,list(zip(yb,zb))
```


```python
## Data Loaders into batches
dl = DataLoader(dset,batch_size=256,shuffle=False)
dl_vd = DataLoader(valid_dset,batch_size=256,shuffle=False) 
# Not sure why True Shuffle is not working even on the validation set. Somehow it gets shuffled wrong

xb,yb = first(dl_vd)
xb.shape, yb.shape
```




    (torch.Size([256, 784]), torch.Size([256, 1]))




```python
type(dl_vd)
```




    fastai.data.load.DataLoader




```python
def init_params() :

    w = torch.randn(28*28).requires_grad_()
    b = torch.randn(1).requires_grad_()
    return (w,b)
```


```python
lr=1
torch.manual_seed(0)
## check what happen if you use other lrs
w,b = init_params()
params=w,b
params[1]
```




    tensor([0.4106], requires_grad=True)




```python
def linear1(tens):
    return tens@w+b   
```


```python
def mnist_loss(prediction,target):
    prediction = prediction.sigmoid()
    return torch.where(target==1,1-prediction,prediction).mean()
## Check what happenes if you use sum (guess: nothing different)
```


```python
mnist_loss(linear1(xb),yb)
```




    tensor(0.2279, grad_fn=<MeanBackward0>)




```python
def calc_grad(xb,yb,model):
    pred = model(xb)
    loss = mnist_loss(pred,yb)
    loss.backward()
```


```python
def train_epoch(dl,model,params):
    for xb,yb in dl:
        calc_grad(xb,yb,model)
        for p in params:
            # print(b,params[1])
            p.data-= p.grad*lr
            # print(b,params[1])
            p.grad.zero_()
            ## p.data is needed otherwise get leaf-variable error
```


```python
train_epoch(dl,linear1,params)
w.grad.mean(),b

```




    (tensor(0.), tensor([0.1381], requires_grad=True))




```python
x,y = first(dl)
x.shape,y.shape
```




    (torch.Size([256, 784]), torch.Size([256, 1]))




```python
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
```


```python
def validate_epoch(dl,model):
    accs = [batch_accuracy(model(xb),yb) for xb,yb in dl]
    return round(torch.stack(accs).mean().item(), 4)

    
```


```python
validate_epoch(dl_vd,linear1)
```




    0.6218




```python
lr=1
torch.manual_seed(0)
## check what happen if you use other lrs
w,b = init_params()
params=w,b
params[1]
```




    tensor([0.4106], requires_grad=True)




```python


for i in range(20):
    train_epoch(dl, linear1, params)
    print(validate_epoch(dl_vd, linear1), end=' ')
```

    0.6218 0.6663 0.8292 0.9007 0.9149 0.9249 0.9306 0.9349 0.9373 0.9396 0.9406 0.9421 0.943 0.9439 0.9453 0.9473 0.9487 0.9501 0.9506 0.9506 

- clean this and then write the variant van the actual, pytorch-versie
- then add non-linearlity
- then test the various questions I had

0.6239 0.6681 0.8348 0.9093 0.9235 0.9333 0.9392 0.9436 0.9456 0.948 0.949 0.9505 0.9515 0.9525 0.9539 0.9559 0.9569 0.9583 0.9588 0.9588

## Doing it with own BASCI OPTIM


```python
linear2 = nn.Linear(28*28,1)
linear2
```




    Linear(in_features=784, out_features=1, bias=True)




```python
w,b = linear2.parameters()
w.shape, b.shape, type(linear2.parameters())
```




    (torch.Size([1, 784]), torch.Size([1]), generator)




```python
b
```




    Parameter containing:
    tensor([-0.0256], requires_grad=True)




```python

class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr
        
    def step(self):
        for p in self.params: p.data -= p.grad *self.lr
            
    def zero_grad(self):
        for p in self.params: p.grad = None #p.grad.zero_()
```


```python
opt = BasicOptim(linear2.parameters(),lr)
```


```python
def train_epoch(dl,model):
    for xb,yb in dl:
        calc_grad(xb,yb,model)
        opt.step()
        opt.zero_grad()
    
```


```python
def train_model(dl,model, no_epochs):
    for i in range(no_epochs):
        train_epoch(dl,model)
        print(validate_epoch(dl_vd,linear2), end=' ')
    
```


```python
lr=1
torch.manual_seed(0)
## check what happen if you use other lrs
opt = BasicOptim(linear2.parameters(),lr)

```


```python
train_model(dl, linear2,20)
```

    0.4932 0.8628 0.8242 0.9116 0.9321 0.9453 0.9551 0.9624 0.9658 0.9673 0.9697 0.9712 0.9731 0.9746 0.9761 0.9765 0.9775 0.9775 0.9785 0.9785 


```python
w.grad,b.grad
```




    (None, None)



## doing it with built in SGD


```python
lr=1
torch.manual_seed(0)
linear2 = nn.Linear(28*28,1)
opt = SGD(linear2.parameters(), lr)
```


```python
type(SGD)
```




    function




```python
def train_model(dl,model, no_epochs):
    for i in range(no_epochs):
        train_epoch(dl,model)
        print(validate_epoch(dl_vd,linear2), end=' ')
    
```


```python
lr=1
torch.manual_seed(0)
## check what happen if you use other lrs
opt = BasicOptim(linear2.parameters(),lr)

```


```python
train_model(dl, linear2,20)
```

    0.4932 0.8843 0.814 0.9087 0.9336 0.9463 0.9555 0.9614 0.9663 0.9673 0.9697 0.9712 0.9741 0.9751 0.9761 0.977 0.9775 0.9775 0.9785 0.9785 

## doing it with fastai


```python
dls = DataLoaders(dl,dl_vd)
```


```python
learn = Learner(dls, nn.Linear(28*28,1),opt_func=SGD, loss_func=mnist_loss,metrics=batch_accuracy)
```


```python
learn.fit(10,lr=lr) ## part which is a for loop of training and validating
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.636650</td>
      <td>0.503657</td>
      <td>0.495584</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.613735</td>
      <td>0.156911</td>
      <td>0.873405</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.222290</td>
      <td>0.198132</td>
      <td>0.818940</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.095358</td>
      <td>0.110542</td>
      <td>0.909715</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.048675</td>
      <td>0.079303</td>
      <td>0.932777</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.030551</td>
      <td>0.063034</td>
      <td>0.945535</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.023174</td>
      <td>0.053108</td>
      <td>0.954858</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.019961</td>
      <td>0.046580</td>
      <td>0.962218</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.018385</td>
      <td>0.042020</td>
      <td>0.965162</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.017474</td>
      <td>0.038670</td>
      <td>0.967615</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## Adding a nonlinearity


```python
def simple_net(xb): 
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```


```python
res = torch.arange(-100,100,10)

res.max(tensor(0))
```




    tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 10, 20, 30, 40, 50, 60, 70, 80, 90])




```python
def init_params2(size): return torch.randn(size).requires_grad_()

```


```python
w1 = init_params2((28*28,30))
b1 = init_params2(1)
w2 = init_params2(30)
b2 = init_params2(1)
```


```python
plot_function(F.relu)
```

    /opt/conda/envs/fastai/lib/python3.8/site-packages/fastbook/__init__.py:55: UserWarning: Not providing a value for linspace's steps is deprecated and will throw a runtime error in a future release. This warning will appear only once per process. (Triggered internally at  /pytorch/aten/src/ATen/native/RangeFactories.cpp:23.)
      x = torch.linspace(min,max)



    
![png](output_68_1.png)
    



```python
simple_net2 = nn.Sequential(
    nn.Linear(28*28,30),
    nn.ReLU(),
    nn.Linear(30,1)
)
```


```python
learn2 = Learner(dls, simple_net2,opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)
```


```python
learn2.fit(40,0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.333953</td>
      <td>0.406201</td>
      <td>0.504416</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.153936</td>
      <td>0.236592</td>
      <td>0.794897</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.084493</td>
      <td>0.117842</td>
      <td>0.909715</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.054969</td>
      <td>0.078843</td>
      <td>0.939647</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.041338</td>
      <td>0.061281</td>
      <td>0.955348</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.034414</td>
      <td>0.051536</td>
      <td>0.964181</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.030461</td>
      <td>0.045446</td>
      <td>0.965162</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.027909</td>
      <td>0.041309</td>
      <td>0.966634</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.026082</td>
      <td>0.038306</td>
      <td>0.968106</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.024675</td>
      <td>0.036016</td>
      <td>0.970559</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.023537</td>
      <td>0.034202</td>
      <td>0.971541</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.022588</td>
      <td>0.032719</td>
      <td>0.973503</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.021780</td>
      <td>0.031472</td>
      <td>0.973503</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.021082</td>
      <td>0.030404</td>
      <td>0.974975</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.020471</td>
      <td>0.029472</td>
      <td>0.974975</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.019929</td>
      <td>0.028649</td>
      <td>0.974975</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.019445</td>
      <td>0.027914</td>
      <td>0.976448</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.019009</td>
      <td>0.027255</td>
      <td>0.977920</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.018613</td>
      <td>0.026659</td>
      <td>0.978410</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.018252</td>
      <td>0.026118</td>
      <td>0.978901</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.017919</td>
      <td>0.025624</td>
      <td>0.978901</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.017612</td>
      <td>0.025171</td>
      <td>0.978901</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.017327</td>
      <td>0.024754</td>
      <td>0.979882</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.017062</td>
      <td>0.024369</td>
      <td>0.980373</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.016814</td>
      <td>0.024014</td>
      <td>0.980864</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.016581</td>
      <td>0.023684</td>
      <td>0.980864</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.016362</td>
      <td>0.023378</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.016156</td>
      <td>0.023094</td>
      <td>0.981845</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.015960</td>
      <td>0.022829</td>
      <td>0.981845</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.015775</td>
      <td>0.022582</td>
      <td>0.981845</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>30</td>
      <td>0.015599</td>
      <td>0.022351</td>
      <td>0.981845</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>31</td>
      <td>0.015431</td>
      <td>0.022134</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>32</td>
      <td>0.015271</td>
      <td>0.021932</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>33</td>
      <td>0.015119</td>
      <td>0.021742</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>34</td>
      <td>0.014972</td>
      <td>0.021563</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>35</td>
      <td>0.014832</td>
      <td>0.021394</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>36</td>
      <td>0.014697</td>
      <td>0.021236</td>
      <td>0.983317</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>37</td>
      <td>0.014567</td>
      <td>0.021085</td>
      <td>0.983317</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>38</td>
      <td>0.014442</td>
      <td>0.020943</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>39</td>
      <td>0.014321</td>
      <td>0.020808</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>



```python
plt.plot(L(learn2.recorder.values).itemgot(2));
```


    
![png](output_72_0.png)
    



```python
## 18 layer model #resnet18
dls = ImageDataLoaders.from_folder(path)
learn = cnn_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, 0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.089402</td>
      <td>0.013574</td>
      <td>0.997056</td>
      <td>00:14</td>
    </tr>
  </tbody>
</table>


- clean up for blogpost
- sexercises
- check on what is different again
- Let's go. to lesson 4.5


```python

```
