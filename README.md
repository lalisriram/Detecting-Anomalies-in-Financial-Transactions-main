
# Detecting Anomalies in Financial Transactions
# video link:- https://drive.google.com/file/d/1Tv9MvbBC6LlT82vjmHLWOdrNXSk8XCQN/view?usp=sharing
#TEAM NUMBER:VH116

#NAME                   #EMAIL
KOPPU ABHINAY        9921004361@klu.ac.in
PERNEEDI SITARAM     9921004560@klu.ac.in
KODAVALUR LALISRIRAM  9921004343@klu.ac.in
POGAKU NOOR AHAMMAD   9921004570@klu.ac.in




#PROBLEM STATEMENT
Detecting Anomalies In Financial Transaction
Our main motto is to find the fraud in the Finanacial Transaction in all over the world transactions.
we need solution to stop the fraud that is happening all over the world.

#ABOUT THE PROJECT

Derived from this observation we distinguish two classes of anomalous journal entries, namely **"global"** and **"local" anomalies** as illustrated in **Figure 2** below:

![image](https://user-images.githubusercontent.com/64821137/231603556-ac7f2d61-14f5-4bc9-804b-859059f4104f.png)

***Global Anomalies***, are financial transactions that exhibit **unusual or rare individual attribute values**. These anomalies usually relate to highly skewed attributes e.g. seldom posting users, rarely used ledgers, or unusual posting times. 

Traditionally "red-flag" tests, performed by auditors during annual audits, are designed to capture those types of anomalies. However, such tests might result in a high volume of false positive alerts due to e.g. regular reverse postings, provisions and year-end adjustments usually associated with a low fraud risk.

***Local Anomalies***, are financial transactions that exhibit an **unusual or rare combination of attribute values** while the individual attribute values occur quite frequently e.g. unusual accounting records. 

This type of anomaly is significantly more difficult to detect since perpetrators intend to disguise their activities trying to imitate a regular behaviour. As a result, such anomalies usually pose a high fraud risk since they might correspond to e.g. misused user accounts, irregular combinations of general ledger accounts and posting keys that don't follow an usual activity pattern.

The objective of this lab is to walk you through a deep learning based methodology that can be used to detect of global and local anomalies in financial datasets. The proposed method is based on the following assumptions: 

>1. the majority of financial transactions recorded within an organizations’ ERP-system relate to regular day-to-day business activities and perpetrators need to deviate from the ”regular” in order to conduct fraud,
>2. such deviating behaviour will be recorded by a very limited number of financial transactions and their respective attribute values or combination of attribute values and we refer to such deviation as "anomaly".

#TECHNICAL IMPLEMENTATION
Concluding from these assumptions we can learn a model of regular journal entries with minimal ”harm” caused by the potential anomalous ones.

In order to detect such anomalies, we will train deep autoencoder networks to learn a compressed but "lossy" model of regular transactions and their underlying posting pattern. Imposing a strong regularization onto the network hidden layers limits the networks' ability to memorize the characteristics of anomalous journal entries. Once the training process is completed, the network will be able to reconstruct regular journal entries, while failing to do so for the anomalous ones.

After completing the lab you should be familiar with:

>1. the basic concepts, intuitions and major building blocks of autoencoder neural networks,
>2. the techniques of pre-processing financial data in order to learn a model of its characteristics,
>3. the application of autoencoder neural networks to detect anomalies in large-scale financial data, and,
>4. the interpretation of the detection results of the networks as well as its reconstruction loss. 

Please note, that this lab is neither a complete nor comprehensive forensic data analysis approach or fraud examination strategy. However, the methodology and code provided in this lab can be modified or adapted to detect anomalous records in a variety of financial datasets. Subsequently, the detected records might serve as a starting point for a more detailed and substantive examination by auditors or compliance personnel.

### Initial Data and Attribute Assessment

The dataset was augmented and renamed the attributes to appear more similar to a real-world dataset that one usually observes in SAP-ERP systems as part of SAP's Finance and Cost controlling (FICO) module. 

below a list of the individual attributes as well as a brief description of their respective semantics:

>- `BELNR`: the accounting document number,
>- `BUKRS`: the company code,
>- `BSCHL`: the posting key,
>- `HKONT`: the posted general ledger account,
>- `PRCTR`: the posted profit center,
>- `WAERS`: the currency key,
>- `KTOSL`: the general ledger account key,
>- `DMBTR`: the amount in local currency,
>- `WRBTR`: the amount in document currency.

Let's also have a closer look into the top 10 rows of the dataset:

<p align="center">
  <img src="https://user-images.githubusercontent.com/64821137/231604111-a92a5690-7cb1-4f08-a860-63f0d445f84b.png" />
</p>

You may also have noticed the attribute `label` in the data. We will use this field throughout to evaluate the quality of our trained models. The field describes the true nature of each individual transaction of either being a **regular** transaction (denoted by `regular`) or an **anomaly** (denoted by `global` and `local`).

### Autoencoder Neural Networks (AENNs)

The objective of this section is to familiarize ourselves with the underlying idea and concepts of building a deep autoencoder neural network (AENN). We will cover the major building blocks and the specific network structure of AENNs as well as an exemplary implementation using the open source machine learning library PyTorch.

### Autoencoder Neural Network Architecture


AENNs or "Replicator Neural Networks" are a variant of general feed-forward neural networks that have been initially introduced by Hinton and Salakhutdinov in [6]. AENNs usually comprise a **symmetrical network architecture** as well as a central hidden layer, referred to as **"latent"** or **"coding" layer**, of lower dimensionality. The design is chosen intentionally since the training objective of an AENN is to reconstruct its input in a "self-supervised" manner. 

below illustrates a schematic view of an autoencoder neural network:

![image](https://user-images.githubusercontent.com/64821137/231604818-312778e7-a652-450f-82a1-ab242fb1a5cf.png)

**Figure:** Schematic view of an autoencoder network comprised of two non-linear mappings (fully connected feed forward neural networks) referred to as encoder $f_\theta: \mathbb{R}^{dx} \mapsto \mathbb{R}^{dz}$ and decoder $g_\theta: \mathbb{R}^{dz} \mapsto \mathbb{R}^{dx}$.

Furthermore, AENNs can be interpreted as "lossy" data **compression algorithms**. They are "lossy" in a sense that the reconstructed outputs will be degraded compared to the original inputs. The difference between the original input $x^i$ and its reconstruction $\hat{x}^i$ is referred to as **reconstruction error**. In general, AENNs encompass three major building blocks:


>   1. an encoding mapping function $f_\theta$, 
>   2. a decoding mapping function $g_\theta$, 
>   3. and a loss function $\mathcal{L_{\theta}}$.

Most commonly the encoder and the decoder mapping functions consist of **several layers of neurons followed by a non-linear function** and shared parameters $\theta$. The encoder mapping $f_\theta(\cdot)$ maps an input vector (e.g. an "one-hot" encoded transaction) $x^i$ to a compressed representation $z^i$ referred to as latent space $Z$. This hidden representation $z^i$ is then mapped back by the decoder $g_\theta(\cdot)$ to a re-constructed vector $\hat{x}^i$ of the original input space (e.g. the re-constructed encoded transaction). Formally, the nonlinear mappings of the encoder- and the decoder-function can be defined by:

<p align="center">
  $f_\theta(x^i) = s(Wx^i + b)$, and $g_\theta(z^i) = s′(W′z^i + d)$
</p>

where $s$ and $s′$ denote non-linear activations with model parameters $\theta = \{W, b, W', d\}$, $W \in \mathbb{R}^{d_x \times d_z}, W' \in \mathbb{R}^{d_z \times d_y}$ are weight matrices and $b \in \mathbb{R}^{dx}$, $d \in \mathbb{R}^{dz}$ are offset bias vectors.

### Autoencoder Neural Network Implementation

Some elements of the encoder network code below should be given particular attention:

>- `self.encoder_Lx`: defines the linear transformation of the layer applied to the incoming input: $Wx + b$.
>- `nn.init.xavier_uniform`: inits the layer weights using a uniform distribution. 
>- `self.encoder_Rx`: defines the non-linear transformation of the layer: $\sigma(\cdot)$.
>- `self.dropout`: randomly zeros some of the elements of the input tensor with probability $p$.

We use **"Leaky ReLUs"** as introduced by Xu et al. to avoid "dying" non-linearities and to speed up training convergence. Leaky ReLUs allow a small gradient even when a particular neuron is not active. In addition, we include the **"drop-out" probability**, which defines the probability rate for each neuron to be set to zero at a forward pass to prevent the network from overfitting. 

Initially, we set the dropout probability of each neuron to $p=0.0$ (0%), meaning that none of the neuron activiations will be set to zero, the default interpretation of the dropout hyperparameter is the probability of training a given node in a layer, where 1.0 means no dropout, and 0.0 means no outputs from the layer.

### Evaluating the Autoencoder Neural Network (AENN) Model

The visualization reveals that the pre-trained model is able to reconstruct the majority of regular journal entries, while failing to do so, for the anomalous ones. As a result, the model reconstruction error can be used to distinguish both "global" anomalies (orange) and "local" anomalies (green) from the regular journal entries (blue).

<p align="center">
  <img src="https://user-images.githubusercontent.com/64821137/231606709-1c1b4aa9-41f4-4e41-b494-a5b2e0ebfd1a.png" />
</p>
![image](https://github.com/lalisriram/Detecting-Anomalies-in-Financial-Transactions-main/assets/163636482/a6340938-2c6e-4c9d-8801-edf608dab181)

#TECHSTACKS USED
MachineLearning,Neural Network,Encoder,Autoencoder.

##How To Run Locally
STEP-1
import os
import sys
from datetime import datetime
import pandas as pd
import random as rd
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from IPython.display import Image, display
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

STEP-2
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The CUDNN backend version: {}'.format(now, torch.backends.cudnn.version()))

STEP-3
USE_CUDA = True
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] The Python version: {}'.format(now, sys.version))

STEP-4
seed_value = 1234
rd.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if (torch.backends.cudnn.version() != None and USE_CUDA == True):
    torch.cuda.manual_seed(seed_value)
    
STEP-5
ori_dataset = pd.read_csv('fraud_dataset_v2.csv')
ori_dataset.head()

STEP-6
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] Transactional dataset of {} rows and {} columns loaded'.format(now, ori_dataset.shape[0], ori_dataset.shape[1]))

STEP-7
ori_dataset.label.value_counts()

STEP-8
print(ori_dataset.columns)
print(ori_dataset.head())
if 'label' in ori_dataset.columns:
    label = ori_dataset.pop('label')
else:
    print("The 'label' column does not exist in ori_dataset.")
    
STEP-9
ori_dataset.head(10)

STEP-10
fig, ax = plt.subplots(1,2)
fig.set_figwidth(20)
g = sns.countplot(x=ori_dataset['BSCHL'], ax=ax[0])
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title('Distribution of BSCHL attribute values')
g = sns.countplot(x=ori_dataset['HKONT'], ax=ax[1])
g.set_xticklabels(g.get_xticklabels(), rotation=90)
g.set_title('Distribution of HKONT attribute values')

STEP-11
categorical_attr_names = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT','WAERS', 'BUKRS']

STEP-12
ori_dataset_categ_transformed = pd.get_dummies(ori_dataset[categorical_attr_names])

STEP-13
fig, ax = plt.subplots(1,2)
fig.set_figwidth(20)
g = sns.distplot(ori_dataset['DMBTR'].tolist(), ax=ax[0])
g.set_title('Distribution of DMBTR amount values')
g = sns.distplot(ori_dataset['WRBTR'].tolist(), ax=ax[1])
g.set_title('Distribution of WRBTR amount values')

STEP-14
import numpy as np
numeric_attr_names = ['DMBTR', 'WRBTR']
numeric_attr = ori_dataset[numeric_attr_names] + 1e-7
numeric_attr = numeric_attr.apply(np.log)
ori_dataset_numeric_attr = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())

STEP-15
numeric_attr_vis = ori_dataset_numeric_attr.copy()
numeric_attr_vis['label'] = label
g = sns.pairplot(data=numeric_attr_vis, vars=numeric_attr_names, hue='label')
g.fig.suptitle('Distribution of DMBTR vs. WRBTR amount values')
g.fig.set_size_inches(15, 5)

STEP-16
ori_subset_transformed = pd.concat([ori_dataset_categ_transformed, ori_dataset_numeric_attr], axis = 1)

STEP-17
ori_subset_transformed.shape

STEP-18
import gc
gc.collect()

STEP-19
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encoder_L1 = nn.Linear(in_features=ori_subset_transformed.shape[1], out_features=3, bias=True) # add linearity
        nn.init.xavier_uniform_(self.encoder_L1.weight) # init weights according to [9]
        self.encoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) # add non-linearity according to [10]
    def forward(self, x):
        x = self.encoder_R1(self.encoder_L1(x)) # don't apply dropout to the AE bottleneck
        return x
        
STEP-20
encoder_train = encoder()
if (torch.backends.cudnn.version() != None and USE_CUDA == True):
  encoder_train = encoder()
  
STEP-21
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] encoder architecture:\n\n{}\n'.format(now, encoder_train))

STEP-22
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.decoder_L1 = nn.Linear(in_features=3, out_features=ori_subset_transformed.shape[1], bias=True) # add linearity
        nn.init.xavier_uniform_(self.decoder_L1.weight)  # init weights according to [9]
        self.decoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) # add non-linearity according to [10]
    def forward(self, x):
        x = self.decoder_R1(self.decoder_L1(x))
        return x
        
STEP-23
decoder_train = decoder()
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
    decoder_train = decoder()
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] decoder architecture:\n\n{}\n'.format(now, decoder_train))

STEP-24
loss_function = nn.BCEWithLogitsLoss(reduction='mean')

STEP-25
learning_rate = 1e-3
encoder_optimizer = torch.optim.Adam(encoder_train.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder_train.parameters(), lr=learning_rate)

STEP-26
num_epochs = 5
mini_batch_size = 128

STEP-27
torch_dataset = torch.from_numpy(ori_subset_transformed.values).float()
dataloader = DataLoader(torch_dataset, batch_size=mini_batch_size, shuffle=True, num_workers=0)
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
    dataloader = DataLoader(torch_dataset, batch_size=mini_batch_size, shuffle=True)
    
STEP-28
losses = []
data = autograd.Variable(torch_dataset)
for epoch in range(num_epochs):
    mini_batch_count = 0
    if(torch.backends.cudnn.version() != None) and (USE_CUDA == True):
        encoder_train
        decoder_train
    encoder_train.train()
    decoder_train.train()
    start_time = datetime.now()
    for mini_batch_data in dataloader:
        mini_batch_count += 1
        mini_batch_torch = autograd.Variable(mini_batch_data)
        z_representation = encoder_train(mini_batch_torch) # encode mini-batch data
        mini_batch_reconstruction = decoder_train(z_representation) # decode mini-batch data
        reconstruction_loss = loss_function(mini_batch_reconstruction, mini_batch_torch)
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        reconstruction_loss.backward()
        decoder_optimizer.step()
        encoder_optimizer.step()
        if mini_batch_count % 1000 == 0:
            mode = 'GPU' if (torch.backends.cudnn.version() != None) and (USE_CUDA == True) else 'CPU'
            now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
            end_time = datetime.now() - start_time
            print('[LOG {}] training status, epoch: [{:04}/{:04}], batch: {:04}, loss: {}, mode: {}, time required: {}'.format(now, (epoch+1), num_epochs, mini_batch_count, np.round(reconstruction_loss.item(), 4), mode, end_time))
            start_time = datetime.now()
    encoder_train.cpu().eval()
    decoder_train.cpu().eval()
    reconstruction = decoder_train(encoder_train(data))
    reconstruction_loss_all = loss_function(reconstruction, data)
    losses.extend([reconstruction_loss_all.item()])
    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    print('[LOG {}] training status, epoch: [{:04}/{:04}], loss: {:.10f}'.format(now, (epoch+1), num_epochs, reconstruction_loss_all.item()))
    encoder_model_name = "ep_{}_encoder_model.pth".format((epoch+1))
    torch.save(encoder_train.state_dict(), os.path.join("/content/drive/MyDrive", encoder_model_name))
    decoder_model_name = "ep_{}_decoder_model.pth".format((epoch+1))
    torch.save(decoder_train.state_dict(), os.path.join("/content/drive/MyDrive", decoder_model_name))
    gc.collect()
    
STEP-29
plt.plot(range(0, len(losses)), losses)
plt.xlabel('[training epoch]')
plt.xlim([0, len(losses)])
plt.ylabel('[reconstruction-error]')
plt.title('AENN training performance')

STEP-30
encoder_model_name = "/content/drive/MyDrive/ep_5_encoder_model.pth"
decoder_model_name = "/content/drive/MyDrive/ep_5_decoder_model.pth"
encoder_eval = encoder()
decoder_eval = decoder()
encoder_eval.load_state_dict(torch.load(os.path.join("models", encoder_model_name)))
decoder_eval.load_state_dict(torch.load(os.path.join("models", decoder_model_name)))

STEP-31
data = autograd.Variable(torch_dataset)
encoder_eval.eval()
decoder_eval.eval()
reconstruction = decoder_eval(encoder_eval(data))

STEP-32
reconstruction_loss_all = loss_function(reconstruction, data)
now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
print('[LOG {}] collected reconstruction loss of: {:06}/{:06} transactions'.format(now, reconstruction.size()[0], reconstruction.size()[0]))
print('[LOG {}] reconstruction loss: {:.10f}'.format(now, reconstruction_loss_all.item()))

STEP-33
reconstruction_loss_transaction = np.zeros(reconstruction.size()[0])
for i in range(0, reconstruction.size()[0]):
    reconstruction_loss_transaction[i] = loss_function(reconstruction[i], data[i]).item()
    if(i % 100000 == 0):
        now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
        print('[LOG {}] collected individual reconstruction loss of: {:06}/{:06} transactions'.format(now, i, reconstruction.size()[0]))

#WHATS NEXT ?
In future we are going to update this project by building a website in the name of anomalies in financial transactions so that people working in banks and 
other sectors can access the transaction data and can conform that the transactions done by the customers is fraud or genuine.

#DECLARATION
We confirm that the project showcased here was either developed entirely during the hackathon or underwent significant updates within the hackathon timeframe. We understand that if any plagiarism from online sources is detected, our project will be disqualified, and our participation in the hackathon will be revoked.
