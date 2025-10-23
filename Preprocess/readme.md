
## 🔁 Reproducibility Pipeline

> This section summarizes the end-to-end data processing flow used to reproduce the results.

### 1) Time Zoning & Behavior Aggregation
- Split timestamps by *Shanghai Time** into *Weekend/Weekday* and four periods (*Early Morning 00:00–05:59*, *Morning 06:00–11:59*, *Afternoon 12:00–17:59*, *Late Night 18:00–23:59*).  
  Aggregate user behaviors to obtain `pv_count / cart_count / fav_count / buy_count` and `buy_yn`.
- **Dataset:** [`dataset_for_prediction.xlsx`](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/f85a24d502a24f85334737841104476a911433d4/Preprocess/dataset_for_prediction.xlsx)  
- **Code:** [`dataset_for_prediction.py`](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/f85a24d502a24f85334737841104476a911433d4/Preprocess/dataset_for_prediction.py)

> ℹ️ *Tip:* Timestamps are converted to **Asia/Shanghai** and floored to the hour to avoid second-level drift.

---

### 2) Dataset Cleaning (Rule-based)
- Remove users with **high browsing but no purchases** and users with **high purchases but low browsing** to reduce noise and extreme behaviors.
- **Dataset:** [`cleaned_dataset.xlsx`](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/f85a24d502a24f85334737841104476a911433d4/Preprocess/cleaned_dataset.xlsx)  
- **Code:** [`miss_data.py`](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/f85a24d502a24f85334737841104476a911433d4/Preprocess/miss_data.py)

---

### 3) Feature Engineering
- Construct descriptive statistics per behavior stream, e.g. `pv_min / pv_max / pv_avg` (and analogs for `cart/fav/buy`), plus downstream features in later steps.
- **Dataset:** [`user_time_with_user_stats.xlsx`](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/f85a24d502a24f85334737841104476a911433d4/Preprocess/user_time_with_user_stats.xlsx)  
- **Code:** [`mean_features.py`](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/f85a24d502a24f85334737841104476a911433d4/Preprocess/mean_features.py)

---

### 4) Final Training Set (Drop Day/Time Fields)
-In line with research objectives and comparative tests, remove day/time related fields to obtain the **final training dataset**.
- **Dataset:** [`no_day1.xlsx`](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/f85a24d502a24f85334737841104476a911433d4/Preprocess/no_day1.xlsx)  
- **Code:** [`pre_dataset.py`](https://github.com/SodaQiu/A_Hierarchical_Feature_Engineering_and_Evaluation_Framework_for_E-commerce_Purchase_Prediction/blob/f85a24d502a24f85334737841104476a911433d4/Preprocess/pre_dataset.py)
