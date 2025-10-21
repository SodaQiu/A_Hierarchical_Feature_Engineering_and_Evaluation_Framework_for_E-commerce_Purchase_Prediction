# A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction

---

This repository is designed to ensure the reproducibility of the article. <br />
You can run the research results through the following steps. <br />
**Note**: This study was conducted with Korean speech data.

---

## Research Content

Individuals with dysarthria exhibit irregular speech patterns depending on the characteristics of their disease, significantly reducing the accuracy of conventional speech recognition systems. Most prior studies have compared only a single disease group or used aggregated data without distinguishing between diseases, failing to adequately analyze disease-specific differences. This study extracted fluency metrics from a Korean dysarthric speech corpus across three disease groups—stroke, cerebral palsy, and peripheral neuropathy—and classified the diseases based on these features. Then, the performance of customized speech recognition models for each disease was evaluated using Weighted Character Error Rate (Weighted-CER). The results showed that the classification accuracy based on fluency metrics reached 99%, and the disease-specific models improved Weighted-CER by up to 18.34 and 1.05 percentage points compared to the Whisper-Small model and a model trained on the entire dataset, respectively. In terms of Weighted-CER, the error rate decreased by up to 15.27 and 1.49 percentage points, respectively. These findings indicate that disease-specific models can meaningfully enhance speech recognition performance for dysarthric speech and highlight the necessity of developing speaker-customized speech recognition systems.

---

## Comparison of utterance samples

| ![image](https://github.com/user-attachments/assets/e6a47b9a-9610-4b42-ab61-f8d7b2b9ce5a) | ![image](https://github.com/user-attachments/assets/47925d2c-3370-4948-b897-6ec08740ee47) |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |

<p align='center'>Fig 1, 2. Disease-specific speech Visualization of waveform+VAD and spectrograms</p>

---

## Reproducibility

<img width="2300" height="1080" alt="ASR_Figure4" src="https://github.com/user-attachments/assets/f7ee6291-e478-4fa2-a202-2ad1f95a8ca3" />
<p align='center'>Fig 3. Overview of our approach</p>

Note: The raw voice data is preprocessed, and feature extraction is performed. The extracted features are used to classify diseases. If there is misclassified voice data, the data is passed to the General Model that is trained to extract the text.

</br>

---

### Research Code and Model Repository

#### Research code

- [Preprocess](Preprocess)
- [Feature Extraction](Feature_Extraction)
- [Preliminary Experiments](Preliminary Experiments)
- [SMOTE_Application](SMOTE_Application)
- [LR_Model_Testing](LR_MODEL_Testing)
- [Other_Model_Testing](Other_Model_Testing)


</br >

#### Fine-Tuned Model

As the STT model before fine-tuning, we used openAI's Whisper-small model. [Whisper](https://github.com/openai/whisper) </br >

- [Stroke](https://huggingface.co/yoona-J/ASR_Whisper_Stroke)
- [Cerebral Palsy](https://huggingface.co/yoona-J/ASR_Whisper_Celebral_Palsy_Aug)
- [Peripheral Neuropathy](https://huggingface.co/yoona-J/ASR_Whisper_Peripheral_Neuropathy)
- [General](https://huggingface.co/yoona-J/ASR_Whisper_Disease_General)

</br >
  
### Preprocessed Training Datasets
Due to AI-Hub's policy, we cannot distribute the original data, so we provide it preprocessed as a log-Mel spectrogram. </br>
The original data can be downloaded through the link below.
[AI-Hub, "Speech recognition data for dysarthria"](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=608)
</br>

- [Stroke Dataset](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Stroke_Dataset)
- [Cerebral Palsy Dataset](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Celebral_Palsy_Dataset_Aug)
- [Peripheral Neuropathy Dataset](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Peripheral_Neuropathy_Dataset)
- [General Dataset](https://huggingface.co/datasets/yoona-J/ASR_Preprocess_Disease_General_Dataset)

</br >

### Utterance Testset

#### Predict Utterance Dataset

- [Text_Preprocess](https://drive.google.com/drive/folders/1nj2i7ATyR_r-g64zfu_cPJrdQniVXv36?usp=drive_link)
- [Audio_Preprocess](https://drive.google.com/drive/folders/1gtUJdO5jkNTziiJcdAayRF_uyqLbpMqF?usp=drive_link)

#### Fluency Analysis Values

We provide feature extraction figures as a file, which also includes speaker information (age, sex, disease). </br>

- [fluency_analysis](Feature_Extraction/Result/fluency_analysis.csv)

</br >

### Frameworks and Libraries Used

- Local Environment (python 3.10.1, pandas 2.2.3, numpy 1.26.4, openpyxl 3.1.5, noisereduce 3.0.3, praat-parselmouth 0.4.5, librosa 0.11.0)
- Google Colab Environment (python 3.11.13, transformers 4.54.0, torch 2.6.0+cu124, numpy 2.0.2, datasets 3.6.0, evaluate 0.4.4, librosa 0.11.0, nlptutti 0.0.0.10, jiwer 4.0.0)

</br >

### Training and Experimental Setup

- Local (NVIDIA GeForce MX150 & Intel® UHD Graphics 620 GPU, Intel Core i7-8565U @ 1.80GHz (4 cores / 8 threads) CPU, 16GB Ram)
- Google Colab (NVIDIA Tesla T4 GPU, 15GB VRAM, CUDA 12.4, Driver 550.54.15)

---

If you have any questions regarding the research, please contact us at the email below. </br>

<a href=mailto:chungyn@hanyang.ac.kr> <img src="https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white&link=mailto:chungyn@hanyang.ac.kr"> </a>

chungyn@hanyang.ac.kr </br>

---
