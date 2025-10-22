# A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction

---

This repository is designed to ensure the reproducibility of the article. <br />
You can run the research results through the following steps. <br />
**Note**: This study was conducted with Korean speech data.

---

## Research Content

With the rapid growth of e-commerce, accurately predicting user purchasing behavior has become essential for enhancing platform competitiveness. However, existing studies often lack a generalizable feature engineering framework that systematically transforms raw behavior logs into high-value predictive signals. To address this gap, this study proposes and validates a three-layer feature engineering framework using real-world data from Alibaba. The framework organizes features into Basic (Basic Features), Conversion (Conversion Rate and Behavior Stability), and Advanced layers (Overall Activity and advanced Interactions). To systematically quantify model performance, this study devised a hierarchical evaluation method that quantifies marginal feature contribution layer by layer. Experimental results demonstrate that this framework improves the F1 score from 0.61 to 0.68, and up to 0.96 in controlled experiments, although this may reflect dataset-specific factors and should be further validated on larger-scale data. Overall, the findings demonstrate the effectiveness of hierarchical feature design and provide a systematic and interpretable approach for user behavior modeling in e-commerce.

---

## Flowchart of Research Processes

| ![image](https://github.com/user-attachments/assets/e6a47b9a-9610-4b42-ab61-f8d7b2b9ce5a) |

<p align='center'>Fig 1. Flowchart of building a user purchase behavior prediction model</p>


![image](https://github.com/user-attachments/assets/47925d2c-3370-4948-b897-6ec08740ee47) |


<p align='center'>Fig 2. Feature hierarchy conceptual diagram illustrates the conceptual flow among feature layers</p>

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
- [Preliminary_Experiments](directory)
- [SMOTE_test](SMOTE_test)
- [LR_Model_train](LR_MODEL_train)
- [Other_Models](Other_Models)


</br >

  
### Preprocessed Training Datasets
The original data can be downloaded through the link below.
[User Behavior Data from Taobao for Recommendation](https://tianchi.aliyun.com/dataset/649?lang=en-us)

</br>

- [Google_Drive, "e-commerce_dataset"](https://docs.google.com/spreadsheets/d/1klqayHnwfpLxYwPlv5sq67ybDQWFGjpd/edit?usp=drive_link&ouid=109035804013317356516&rtpof=true&sd=true)
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
- 

</br >

### Training and Experimental Setup

- Local (NVIDIA GeForce MX150 & Intel® UHD Graphics 620 GPU, Intel Core i7-8565U @ 1.80GHz (4 cores / 8 threads) CPU, 16GB Ram)
- Google Colab (NVIDIA Tesla T4 GPU, 15GB VRAM, CUDA 12.4, Driver 550.54.15)

---

If you have any questions regarding the research, please contact us at the email below. </br>


soda0808@hanyang.ac.kr </br>

---
