# A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction

---

This repository is designed to ensure the reproducibility of the article. <br />
You can run the research results through the following steps. <br />
**Note**: This study was conducted with Korean speech data.

---

## Research Content

With the rapid growth of e-commerce, accurately predicting user purchasing behavior has become essential for enhancing platform competitiveness. However, existing studies often lack a generalizable feature engineering framework that systematically transforms raw behavior logs into high-value predictive signals. To address this gap, this study proposes and validates a three-layer feature engineering framework using real-world data from Alibaba. The framework organizes features into Basic (Basic Features), Conversion (Conversion Rate and Behavior Stability), and Advanced layers (Overall Activity and Advanced Interactions). To systematically quantify model performance, this study devised a hierarchical evaluation method that quantifies marginal feature contribution layer by layer. Experimental results show that the framework improves the F1 score of the LR model from 0.61 to 0.96, but this may reflect dataset-specific factors and needs further verification on larger-scale data. Overall, the findings demonstrate the effectiveness of hierarchical feature design and provide a systematic and interpretable approach for user behavior modeling in e-commerce.

---

## Flowchart of Research Processes

| ![image](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/43cf864e362e220935049afe2590ff6258970612/Feature%20Extraction/Flowchart%20of%20building%20a%20user%20purchase%20behavior%20prediction%20model.png)

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

### Research Code 

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

- [Final_dataset](https://docs.google.com/spreadsheets/d/1klqayHnwfpLxYwPlv5sq67ybDQWFGjpd/edit?usp=drive_link&ouid=109035804013317356516&rtpof=true&sd=true)
- [Cleaned_dataset](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Preprocess/cleaned_dataset.xlsx)
- [Dataset_for_prediction](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Preprocess/dataset_for_prediction.xlsx)
- [User_time_with_user_stats](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/38b4ae554083bef643497ff713561d0057a99fce/Preprocess/user_time_with_user_stats.xlsx)

</br >

### Methodology

This framework consists of three hierarchical feature layers:

- **Base Layer**: Captures raw user activity metrics (e.g., pv_count, cart_count, fav_count).

- **Conversion Layer**: Focuses on conversion rate metrics and stability (e.g., cart_stability, fav_to_pv_count_rate).

- **Advanced Layer**: Represents overall activity and high-level interaction features (e.g., pv_cart_interaction, ..._ratio).

- [Smote_Comparison](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/ab12ddcef9e5b6dcf92ef76fe798a7270d5da96f/SMOTE_test/smote_visualization.py)

Note: To address class imbalance and ensure robust evaluation, the Synthetic Minority Oversampling Technique (SMOTE) and stratified 5-fold cross-validation are applied during training.



</br>

### Results and Discussion

- The hierarchical feature framework improved model performance across of XGB and CatBoots.  
- Logistic Regression F1-score improved from **0.6126 → 0.9624**, validating the effectiveness of the proposed features.  
- CatBoost achieved the best performance with **F1 = 0.9826**, demonstrating strong capability in capturing non-linear interactions.  
- The results confirm the robustness and scalability of the hierarchical feature engineering design. </br>

- [CatBoost_code](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/ab12ddcef9e5b6dcf92ef76fe798a7270d5da96f/Other_Models/Catboost_1.py)

</br >

### Reproducibility Steps

If you don't want to start with data preprocessing, download the following dataset.[Final_dataset](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/e94a76deb7f4e3c6bb1f9d77dcc2157282e041c6/Preprocess/no_day1.xlsx)
  
- First, consider the purpose of the study and conduct a simple comparative test, then consider removing the time feature.
[no_day_and_time](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/78b54449986d89af8b61d20cd51bb179f6555cc0/Feature_Extraction/no_time_date.py)

- Experiment with basic features in the base model. See the directory documentation for details.
[Stacking_SMOTE_XGB_LR_RF](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/78b54449986d89af8b61d20cd51bb179f6555cc0/directory/Stacking_SMOTE_XGB_LR_RF.py)

- The third step is to conduct a SMOTE comparative experiment, taking LR and XGB as benchmarks.
[Smote_LR_XGB_test](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/c2c874da8261176314210c3fec2d249b096401a6/SMOTE_test/Smote_LR_XGB_test.py)

- The effectiveness of feature engineering is verified through layered experiments based on the LR model.
[LR_Advanced](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/2385f0a525e9735a40caff523b3731527d33f18a/LR_Model_train/LR_Advanced.py)

- Apply this feature engineering to the other four models to see the results.

[XGB](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/2385f0a525e9735a40caff523b3731527d33f18a/Other_Models/XGBoost_test.py)

[RF](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/2385f0a525e9735a40caff523b3731527d33f18a/Other_Models/RandomForest_test.py)

[CatBoost](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/2385f0a525e9735a40caff523b3731527d33f18a/Other_Models/Catboost_1.py)

[DT](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/2385f0a525e9735a40caff523b3731527d33f18a/Other_Models/DT_test.py)


### Frameworks and Libraries Used

- Local Environment: (python 3.12.4, pandas 2.2.2, numpy 1.26.4, matplotlib 3.8.4, seaborn 0.13.2, scikit-lear 1.4.2, imblearn 0.12.3)
- Models: (XGB 3.0.5, Catboost 1.2.8, DT 1.4.2, RF 1.4.2)

</br >

### Training and Experimental Setup

- Local: AMD Radeon 680M GPU,AMD Ryzen ™7-6800H @32GHz(8cores/16threads) CPU,16GB Ram

---

If you have any questions regarding the research, please contact us at the email below. </br>


soda0808@hanyang.ac.kr </br>

---
