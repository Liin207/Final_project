# <center>Dự Báo Giá Của Năm Đồng Tiền Mã Hoá Sử Dụng Mô Hình Machine Learning và Deep Learning</center>

Đồ án sử dụng ba mô hình: **ARIMA** (base model), **ARIMA - GARCH** và **LSTM** để dự báo giá cả trong tương lai của năm đồng tiền mã hoá lớn: **Bitcoin, Ethereum, Binance Coin, XRP** và **Dogecoin**.

**Dữ liệu** là mức giá giao dịch theo ngày, thu thập từ năm 2017 đến năm 2024, được lấy nguồn từ Kaggle.

**Biến mục tiêu**: Giá đóng cửa (**Close**)

**Thời gian dự báo**: 30 ngày, từ 30 - 11 - 2024 đến 29 - 12 - 2024

**Độ chính xác của mô hình** được đánh giá bằng chỉ số **MAPE**. Mô hình đạt yêu cầu là mô hình có giá trị MAPE nhỏ hơn hoặc bằng **2.34%**.
Dựa trên kết quả đánh giá, mô hình có performance tốt nhất sẽ được lựa chọn để cải tiến nhằm tăng độ chính xác cho dự báo.

## Mục Lục

- [Dự Báo Giá Của Năm Đồng Tiền Mã Hoá Sử Dụng Mô Hình Machine Learning và Deep Learning](#dự-báo-giá-của-năm-đồng-tiền-mã-hoá-sử-dụng-mô-hình-machine-learning-và-deep-learning)
  - [Mục Lục](#mục-lục)
  - [Chuẩn Bị Dữ Liệu và Lập Mô Hình](#chuẩn-bị-dữ-liệu-và-lập-mô-hình)
    - [Mô hình ARIMA](#mô-hình-arima)
    - [Mô hình ARIMA - GARCH](#mô-hình-arima---garch)
    - [Mô hình LSTM](#mô-hình-lstm)
    - [Đánh Giá Kết Quả](#đánh-giá-kết-quả)
  - [Cải Tiến Mô Hình](#cải-tiến-mô-hình)
    - [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
    - [Cấu trúc mô hình](#cấu-trúc-mô-hình)
    - [Kết quả](#kết-quả)
  - [Dự báo](#dự-báo)
  - [Kết Luận](#kết-luận)
  - [Files đính kèm](#files-đính-kèm)
  - [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

## Chuẩn Bị Dữ Liệu và Lập Mô Hình

### Mô hình ARIMA

**1. Chuẩn bị dữ liệu**

- **Phân rã dữ liệu theo mùa:** Kết luận dữ liệu không có tính mùa vụ

- **Kiểm định tính dừng ADF test**:
  - XRP có tính dừng tại chuỗi thời gian gốc
  - Bitcoin, Ethereum, Binance Coin và Dogecoin có tính dừng tại sai phân bậc 1
  
- **Biến đổi dữ liệu**: Biến đổi logarit

- **Tách bộ dữ liệu huấn luyện - kiểm tra**: Tỷ lệ training set/ testing set là 0.7/0.3

**2. Lựa chọn bộ tham số tối ưu**

- Bộ tham số **(p, d, q)** được lựa chọn dựa trên *ADF Test* và `auto_arima()`
<p align="center">
  <img <img src="./Images/Arima_para.png" alt="Arima_para" width="600"/>
</p>

**3. Dự báo trên bộ dữ liệu Test data**

<p align="center">
  <img <img src="./Images/Arima_perform.png" alt="Arima_perform" width="600"/>
</p>

- Giá dự báo đi ngang.

- Không phản ánh được dao động và xu hướng giá trong thực tế.

- Khoảng tin cậy tăng nhanh theo thời gian.

### Mô hình ARIMA - GARCH

**1. Chuẩn bị dữ liệu**

Sử dụng bộ dữ liệu tương tự mô hình ARIMA ở trên

**2. Lựa chọn bộ tham số tối ưu**

- Bộ tham số **(p, d, q)** cho mô hình ARIMA được lựa chọn dựa trên *ADF Test* và `forecast.auto_arima()`
- Bộ tham số **(p, q)** cho mô hình GARCH là *(1, 1)*

<p align="center">
  <img <img src="./Images/Arga_para.png" alt="Arga_para" width="600"/>
</p>

**3. Dự báo trên bộ dữ liệu Test data**

<p align="center">
  <img <img src="./Images/Arga_perform.png" alt="Arga_perform" width="600"/>
</p>

- Đối với Bitcoin và Ethereum:
  
  Dự báo được xu hướng tổng thể nhưng không nắm bắt được biến động chi tiết.

- Đối với Binance:
  
  Không dự báo tốt các biến động giảm giá

- Đối với XRP và Dogecoin
  
  Dự báo xu hướng ngược chiều/ đi ngang so với thực tế

  Khoảng tin cậy rộng, sai số lớn

### Mô hình LSTM

**1. Chuẩn bị dữ liệu**

- **Chuyển dữ liệu về dạng *Sliding Window*** với window là giá đóng cửa tại 15 ngày trước đó

- **Chuẩn hóa dữ liệu** sử dụng hàm `MinMaxScaler()`. Riêng với Dogecoin thực hiện thêm bước logarit

- **Tách bộ dữ liệu huấn luyện - kiểm tra**: Tỷ lệ training set/ testing set là 0.7/0.3

**2. Cấu Trúc Mô Hình**

<p align="center">
  <img <img src="./Images/LSTM.png" alt="LSTM" width="300"/>
</p>

**3. Dự báo trên bộ dữ liệu Test data**

<p align="center">
  <img <img src="./Images/LSTM_perform.png" alt="LSTM_perform" width="600"/>
</p>

Giá dự báo phản ánh đúng dao động và xu hướng giá thực tế.

Có độ trễ trong dự báo, đặc biệt đối với những giai đoạn biến động lớn (điển hình là Dogecoin)

### Đánh Giá Kết Quả

Thực hiện đánh giá mô hình trên bộ dữ liệu kiểm tra. Kết quả mô hình như sau:

<p align="center">
  <img <img src="./Images/Step30.png" alt="Step30" width="450"/>
</p>

<p align="center">
  <img <img src="./Images/Testdata.png" alt="Testdata" width="450"/>
</p>

**Kết Luận**

Mô hình LSTM là mô hình có performance tốt nhất và được lựa chọn làm mô hình để cải tiến.

Mô hình LSTM có thể dự đoán xu hướng chung và các giá trị đỉnh/ đáy của thị trường với độ chính xác cao. Mô hình có thể học những pattern phức tạp và phi tuyến của thị trường tiền mã hoá và đưa ra dự đoán tốt trong ngắn hạn và dài hạn

MAPE tại step 30: 2.86% > 2.34% => Mô hình chưa đáp ứng được độ chính xác yêu cầu. Cần cải tiến thêm

## Cải Tiến Mô Hình

### Chuẩn bị dữ liệu

Bổ sung 6 feature: **SMA60, SMA120, SMA220, Bollinger_Upper, Bollinger_Lower, Return**

### Cấu trúc mô hình
<p align="center">
  <img <img src="./Images/GRU-LSTM.png" alt="GRU-LSTM" width="450"/>
</p>

### Kết quả
<p align="center">
  <img <img src="./Images/Performance_optmodel.png" alt="Performance_optmodel" width="600"/>
</p>

<p align="center">
  <img <img src="./Images/GRU-LSTM_perform.png" alt="GRU-LSTM_perform" width="600"/>
</p>

Giá trị MAPE tại tất cả các chuỗi thời gian đều được cải thiện đáng kể

Mô hình GRU – LSTM đạt được độ chính xác cao khi dự báo giá trong ngắn hạn và dài hạn.

MAPE tại step 30 của Bitcoin là **1.42% < 2.34%** => Mô hình đã đáp ứng được tiêu chí về chất lượng.

## Dự báo

Sử dụng mô hình kết hợp **GRU - LSTM** để dự báo giá của 5 đồng tiền số trong 30 ngày tiếp theo.

<p align="center">
  <img <img src="./Images/Predict.png" alt="Predict" width="600"/>
</p>

## Kết Luận

Thị trường tiền mã hoá là một thị trường tài chính có biên độ dao động lớn và tần suất biến động cao.

Đồ án đã chỉ ra rằng mô hình ARIMA và ARIMA - GARCH có sự hạn chế khi dự báo giá cả trên thị trường này. Cả hai mô hình đều không nắm bắt được xu hướng tổng thể tốt. Mức độ không chắc chắn trong dự báo cao, đặc biệt với XRP và Dogecoin là hai đồng tiền chịu ảnh hưởng mạnh của các yếu tố phi thị trường.

Mô hình LSTM hiệu quả trong việc nắm bắt được những pattern phức tạp với độ trễ nhỏ so với giá thực tế. Độ chính xác của dự báo cao cả trong ngắn hạn và dài hạn.

Sự kết hợp của mô hình GRU và LSTM đã đem lại độ chính xác của dự báo cao nhất. Mô hình kết hợp cải thiện độ trễ và khoảng tin cậy trong dự báo một cách đáng kể.

**Hướng nghiên cứu trong tương lai**

Trong tương lai, hướng nghiên cứu có thể mở rộng theo các tiêu chí:

- Về Dữ Liệu
  
  - Thêm dữ liệu về các yếu tố bên ngoài: Chỉ số kinh tế vĩ mô, chính sách tiền tệ
  
  - Thêm dữ liệu về phân tích cảm xúc (Sentiment Analysis)
  
- Về Mô Hình

  - Kết hợp mô hình ARIMA – LSTM – GRU
  
  - Áp dụng mô hình khác (PELT, Transformer, … )
  
  - Xây dựng mô hình riêng cho từng đồng tiền số
  
- Về Phạm vi dự báo
  
  - Chia nhỏ dữ liệu thành nhiều thời kỳ khác nhau để dự báo
  
  - Dự báo cho quãng thời gian dài hơn

## Files đính kèm

| Folder |File name         | Purpose |
|------|----------------------|------|
|Code|`Crypto_Prices_Prediction.ipynb`| Notebook chứa code cho quá trình xử lý dữ liệu, lựa chọn tham số, huấn luyện mô hình và đánh giá, dự báo |
| |`helps_function.py`| Chứa hàm hỗ trợ cho đánh giá mô hình |
|Dataset|`Crypto_Currency_From_2017_To_2024.csv`| File chứa dữ liệu sử dụng trong đồ án |
|Images|| Chứa biểu đồ, đồ thị sử dụng trong đồ án |
|Report|`[PDF]Research_Report.pdf`| File báo cáo dưới dạng PDF|
||`[PPT]Research_Report.pptx`| File báo cáo dưới dạng PPT|


## Tài Liệu Tham Khảo

1. Comparative study of Bitcoin price prediction using WaveNets, Recurrent Neural Networks and other Machine Learning Methods - Leonardo Felizardo; Roberth Oliveira; Emilio Del-Moral-Hernandez; Fabio Cozman - 2019

2. Comparison of ARIMA Time Series Model and LSTM Deep Learning Algorithm for Bitcoin Price Forecasting - Karakoyun, E. Ş. & Çıbıkdiken, A. O. - 2018

3. Bitcoin Price Trend Prediction Using Deep Neural Network  - Hashem Fekry Nematallah1 , Ahmed Ahmed Hesham Sedky2 , Khaled Mohamed Mahar - 2022

4. The Future of Bitcoin Price Predictions Integrating Deep Learning and the Hybrid Model Method - Belalova Guzalxon∗, Mannanova Shakhida and Karimov Botir. - 2023
