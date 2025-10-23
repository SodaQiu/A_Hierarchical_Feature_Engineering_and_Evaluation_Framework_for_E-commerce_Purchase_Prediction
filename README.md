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

| ![image](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Feature%20Extraction/Flowchart%20of%20building%20a%20user%20purchase%20behavior%20prediction%20model.jpg)

<p align='center'>Fig 1. Flowchart of building a user purchase behavior prediction model</p>


| ![image](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Feature%20Extraction/Feature%20hierarchy%20conceptual%20diagram%20illustrates%20the%20conceptual%20flow%20among%20feature%20layers.png)


<p align='center'>Fig 2. Feature hierarchy conceptual diagram illustrates the conceptual flow among feature layers</p>

---

## Comparison of Research Model Results

<p align="center">
  <img src="https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Feature%20Extraction/lr_feature_importance.png?raw=true" width="900"/>
  <br>
  <em>Fig 3. Top 20 Important Features (Logistic Regression with RFE-based)</em>
</p>

<p align="center">
  <img src="https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Other_Models/catboost_feature_importance.png?raw=true" width="900" alt="Fig 4. Top 20 Important Features (CatBoost on RFE-based Features)" />
  <br>
  <em>Fig 4. Top 20 Important Features (CatBoost on RFE-based Features)</em>
</p>


Note: The raw user behavior logs are preprocessed, and feature engineering is performed to construct multi-layer features. The extracted features are then used to predict user purchase behavior.

</br>

---

### Research Code and Model Repository

#### Research code

- [Preprocess](Preprocess)
- [Feature Extraction](Feature_Extraction)
- [Preliminary_Experiments](directory)
- [SMOTE_test](SMOTE_test)
- [LR_Model_train](LR_Model_train)
- [Other_Models](Other_Models)


</br >

  
### Preprocessed Training Datasets
The original data can be downloaded through the link below.
[User Behavior Data from Taobao for Recommendation](https://tianchi.aliyun.com/dataset/649?lang=en-us)

</br>

- [Final_dataset, "e-commerce_dataset"](https://docs.google.com/spreadsheets/d/1klqayHnwfpLxYwPlv5sq67ybDQWFGjpd/edit?usp=drive_link&ouid=109035804013317356516&rtpof=true&sd=true)
- [cleaned_dataset](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Preprocess/cleaned_dataset.xlsx)
- [dataset_for_prediction](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Preprocess/dataset_for_prediction.xlsx)
- [user_time_with_user_stats](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Preprocess/user_time_with_user_stats.xlsx)

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

- Local Environment (python 3.12.4, pandas 2.2.2, numpy 1.26.4, matplotlib 3.8.4, seaborn 0.13.2, scikit-lear 1.4.2, imblearn 0.12.3)
- Model (XGB 3.0.5, Catboost 1.2.8, DT 1.4.2, RF 1.4.2)

</br >

### Training and Experimental Setup

- Local (AMD Radeon 680M GPU,AMD Ryzen ™7-6800H @32GHz(8cores/16threads)CPU,16GB Ram)

---

If you have any questions regarding the research, please contact us at the email below. </br>


soda0808@hanyang.ac.kr </br>

---
