# DDASR
This is the code reproduce repository for the paper [DDASR: Deep Diverse API Sequence Recommendation]
## Dependency
* python==3.8.0
* pytorch==1.10.2+cu113
* numpy==1.22.2

## File Structure
* Seq2Seq.py: the Seq2Seq model
* Encoder.py: query Encoder
* Decoder.py: API Sequence Decoder
* Evaluate.py: eval the model
* LossLongtail.py: loss function
* data_loader.py: data loader
* Metrics.py: evaluation metrics
* main.py: you can run this file to train the model
## Dataset
* For the original Java dataset, you can download from [https://github.com/huxd/deepAPI].
* For the Python dataset, you can download from [https://github.com/hapsby/deepAPIRevisited].
* For the diverse Java dataset, since the dataset is quite large, I have to upload it using Google Drive. Please download the full package using the following link:
[https://drive.google.com/drive/folders/16c2ZbXr2N2Q_v8fjvLBdUWh2pVQQZhng?usp=sharing]

## Architectures
### RNN encoder-decoder
*BiLSTM is used as the encoder and GRU is used as the decoder.
### Transformer encoder-decoder
*Transformer with six layers is used as the encoder end the decoder.
### LLM encoder-decoder
We utilize five recent LLMs as the encoder and Tansformer with six layers as the decoder.
* CodeBERT: microsoft/codebert-base [https://huggingface.co/microsoft/codebert-base]
* GraphCodeBERT: microsoft/graphcodebert-base [https://huggingface.co/microsoft/graphcodebert-base]
* PLBART: uclanlp/plbart-base [https://huggingface.co/uclanlp/plbart-base]
* CodeT5: Salesforce/codet5-base [https://huggingface.co/Salesforce/codet5-base]
* UniXcoder: microsoft/unixcoder-base [https://huggingface.co/microsoft/unixcoder-base]

## Competing Models
* DeepAPI
the repository of DeepAPI [https://github.com/huxd/deepAPI]
* BIKER
the repository of BIKER [https://github.com/tkdsheep/BIKER-ASE2018]
* CodeBERT
the repository of CodeBERT [https://github.com/hapsby/deepAPIRevisited]
* CodeTrans
the repository of CodeTrans [https://github.com/agemagician/CodeTrans]

