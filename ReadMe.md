# Ranking-based Convolutional Neural Network Models for Peptide-MHC Binding Prediction

This is the implementation of our paper https://arxiv.org/pdf/2012.02840.pdf



## Requirements

Operating systems: Red Hat Enterprise Linux (RHEL) 7.7

- python==3.6.12
- scikit-learn==0.22.1
- pytorch==1.5.1
- scipy==1.4.1



## Training

To train our model,  run

```
python ./src/baseline.py -m [model name] -ls [loss name] -l [fully connected layers] -b [batch size] -f [filter number] -k [filter size] -ft [feature type]
```

<code>-m</code>: specifies the name of model (can be ConvModel or SpannyConvModel)

<code>-ls</code>: specifies the name of loss (can be "HingeLoss1", "HingeLoss2", "HingeLoss3", "MeanSquare")

<code>-l</code>: specifies the fully connected layers (e.x., "[64]" for one single FC layer with 64 hidden units, and "[64, 8]" for two FC layers with 64 and 8 hidden units, respectively.)

<code>-b</code>: specifies the size of batch during the training

<code>-f</code>: specifies the numer of filters in ConvModel and SpannyConvModel.

<code>-k</code>: specifies the size of filters.

<code>-ft</code>: specifies the feature type used by the model (can be "Blosum", "Learned", "One-hot", and their combinations connected with "_", for example, "Blosum_Learned".)



## Note

We haven't organized/cleaned the code until now. Current version can be run successfully.  A neat and clean version will be released later.

