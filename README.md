This repository adds to our research titled "Innovative Approaches to Multivariate Time Series Anomaly Detection: Stacked Transformers and Learnable Embedding". This is a revamped version of the code used to generate the paper's results for simplicity of usage. Follow the methods outlined below to replicate each cell in the results table. 

Installation
This code needs Python-3.7 or higher.

pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt

Now go to the following path /usr/local/lib/python3.10/dist-packages/torch/nn/modules/transformer.py and at line 387 remove the argument “is_causal = is_causal”
It should look like this output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask_for_layers)

Data Preprocessing
You can preprocess the data using the following command:
python preprocess.py --dataset <dataset>

Here, the <dataset> can be either of the following:  'MSL', 'SMAP' or 'SMD'

Train the model:
<!-- The procedure is provided in the code.ipynb file for all the requirement dependencies and steps to run the code  -->
To train the model run the following command:
python train.py --dataset <dataset> --lookback <lookback> --epochs <epochs> --use_gatv2 False --arch <architecture>

where <dataset> can either be 'MSL', 'SMAP' or 'SMD'
and <architecture> specifies the architecture of the Transformer which is being used.

Exaample: <!python train.py --dataset MSL --lookback 30 --epochs 5 --use_gatv2 False --arch 1-1-1>

