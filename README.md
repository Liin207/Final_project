# <center>Dự Báo Giá Của Năm Đồng Tiền Mã Hoá Sử Dụng Mô Hình Machine Learning và Deep Learning</center>

Đồ án sử dụng ba mô hình: **ARIMA** (base model), **ARIMA - GARCH** và **LSTM** để dự báo giá cả trong tương lai của năm đồng tiền mã hoá lớn: **Bitcoin, Ethereum, Binance Coin, XRP** và **Dogecoin**.

**Dữ liệu** là mức giá giao dịch theo ngày, thu thập từ 2017 đến 2024, được lấy nguồn từ Kaggle.

**Độ chính xác của mô hình** được đánh giá bằng chỉ số **MAPE**. Mô hình đạt yêu cầu là mô hình có giá trị MAPE nhỏ hơn hoặc bằng **2.34%**.
Dựa trên kết quả đánh giá, mô hình có performance tốt nhất sẽ được lựa chọn để cải tiến nhằm tăng độ chính xác cho dự báo.

## Mục Lục
- [Dự Báo Giá Của Năm Đồng Tiền Mã Hoá Sử Dụng Mô Hình Machine Learning và Deep Learning](#dự-báo-giá-của-năm-đồng-tiền-mã-hoá-sử-dụng-mô-hình-machine-learning-và-deep-learning)
  - [Mục Lục](#mục-lục)
  - [Lập Mô Hình](#lập-mô-hình)
    - [Mô hình ARIMA](#mô-hình-arima)
    - [Mô hình ARIMA - GARCH](#mô-hình-arima---garch)
    - [Mô hình LSTM](#mô-hình-lstm)
    - [Đánh Giá Kết Quả](#đánh-giá-kết-quả)
  - [Cải Thiện Mô Hình](#cải-thiện-mô-hình)
    - [Tiền xử lý dữ liệu](#tiền-xử-lý-dữ-liệu)
    - [Cấu trúc mô hình](#cấu-trúc-mô-hình)
    - [Kết quả](#kết-quả)
  - [Dự báo](#dự-báo)
  - [Kết Luận](#kết-luận)
  - [Files đính kèm](#files-đính-kèm)
  - [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)


## Lập Mô Hình
### Mô hình ARIMA
- **Phân rã dữ liệu theo mùa:** Kết luận dữ liệu không có tính mùa vụ
- **Kiểm định tính dừng** ADF test cho chuỗi thời gian:
  - XRP có tính dừng tại chuỗi thời gian gốc
  - Bitcoin, Ethereum, Binance Coin và Dogecoin có tính dừng tại sai phân bậc 1
- Dữ liệu sau đó được **biến đổi logarit** và **tách làm dữ liệu huấn luyện - kiểm tra** với tỷ lệ 0.7/0.3
- Bộ tham số **(p, d, q)** được lựa chọn dựa trên *ADF Test* và `auto_arima()`
### Mô hình ARIMA - GARCH
- Sử dụng bộ dữ liệu tương tự mô hình ARIMA ở trên
- Bộ tham số **(p, d, q)** cho mô hình ARIMA được lựa chọn dựa trên *ADF Test* và `forecast.auto_arima()`
- Bộ tham số **(p, q)** cho mô hình GARCH là *(1, 1)*
### Mô hình LSTM
- Tiền xử lý dữ liệu
  - Dữ liệu được chuyển về dạng Sliding Window với window là giá đóng cửa tại 15 ngày trước đó
  - Dữ liệu được biến đổi áp dụng hàm MinMaxScaler(), riêng với Dogecoin thực hiện thêm bước logarit
  - Tách bộ dữ liệu huấn luyện - kiểm tra với tỷ lệ 0.7/ 0.3
- Cấu trúc mô hình
<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Model_Structure.png" width="450"/>
</p>

### Đánh Giá Kết Quả
Sau khi thực hiện đánh giá mô hình trên bộ dữ liệu kiểm tra với các step = 15, 30, 120 và trên toàn bộ bộ dữ liệu, kết quả mô hình như sau:
<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Model_Structure.png" width="450"/>
</p>

Mô hình được lựa chọn để cải tiến là mô hình LSTM.
## Cải Thiện Mô Hình
### Tiền xử lý dữ liệu
Bổ sung 6 feature: **SMA60, SMA120, SMA220, Bollinger_Upper, Bollinger_Lower, Return**
### Cấu trúc mô hình
<p align="center">
  <img src="https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers/blob/main/images/Model_Structure.png" width="450"/>
</p>

### Kết quả
  - Giá trị MAPE tại tất cả các chuỗi thời gian đều được cải thiện đáng kể
  - Giá trị MAPE trên chuỗi dữ liệu Bitcoin tại horizon = 30 là **1.34% < 2.34%**, đạt yêu cầu đề ra
## Dự báo
Sử dụng mô hình kết hợp **GRU - LSTM** để dự báo giá của 5 đồng tiền số trong 30 ngày tiếp theo:

## Kết Luận
Đồ án đã chỉ ra rằng sự kết hợp của mô hình GRU - LSTM đã đem lại độ chính xác của dự báo cao nhất.
Hướng nghiên cứu trong tương lai:
## Files đính kèm

| Folder |File name         | Purpose |
|------|----------------------|------|
|code|`bitcoin_price_prediction.ipynb`| Notebook which includes all data processing, training, and inference |
| |`bitcoin_price_prediction_optuna.ipynb`| Optuna hyperparameters tuning |
|data|`okex_btcusdt_kline_1m.csv.zip`| Zip file containing the data we used in this project |
|images|`Data_Separation.png`| Image that shows our train-validation-tets split |
| |`Model_Structure.png`| Image that shows our model architecture |
| |`Optuna_Result.jpeg`| Image that shows the importance of the Hyperparameters produced by Optuna  |
| |`Test_Prediction.png`| Image that shows our result on the test set |
| |`Test_Presiction_Zoom_In.png`| Image that shows our result on the test set - zoomed-in|
| |`presentation_preview.gif`| Gif showing preview of the project presentation|

## Tài Liệu Tham Khảo
