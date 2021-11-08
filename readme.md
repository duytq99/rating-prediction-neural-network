# MOVIES RATINGS PREDICTION USING NEURAL NETWORK AND APPICATION IN RECOMMENDATION SYSTEM
# Dự đoán đánh giá phim sử dụng mạng nơ-ron và ứng dụng vào hệ gợi ý
<details> 
<summary>Mục lục</summary>

  - [1. Tổng quan](#1-tổng-quan)
    - [Giới thiệu](#giới-thiệu)
    - [Cấu trúc project](#cấu-trúc-project)
  - [2. Cài đặt môi trường và thư viện cần thiết](#2-cài-đặt-môi-trường-và-thư-viện-cần-thiết)
  - [3. Cách thực hiện demo](#3-cách-thực-hiện-demo)
  - [4. Đánh giá](#4-đánh-giá)
  - [5. Thành viên](#5-thành-viên)
  - [6. Tài liệu tham khảo](#6-tài-liệu-tham-khảo)

</details>

## 1. Tổng quan
### Giới thiệu
#### Bài toán đặt ra
#### Tập dữ liệu MovieLens20M
#### Mô hình mạng nơ-ron 

![Mô hình mạng nơ-ron đề xuất](/images/model.png)

What your application does,
Why you used the technologies you used,
Some of the challenges you faced and features you hope to implement in the future.
### Cấu trúc project

## 2. Cài đặt môi trường và thư viện cần thiết
Project này được cài đặt trên môi trưởng ảo được tạo bởi `miniconda`. Lựa chọn phiên bản phù hợp và cài đặt `miniconda` tại [link](https://docs.conda.io/en/latest/miniconda.html).

Khởi tạo môi trường ảo sử dụng `miniconda` với phiên bản python là 3.7:
> `conda create --name <ml-project> python==3.7`

Các thư viện sử dụng trong project này được liệt kê ở file [requirements.txt](#requirements.txt), bao gồm:
- `keras==2.6.0, tensorflow==2.7.0, tensorflow_cpu==2.6.1`: xây dựng mô hình mạng nơ-ron
- `numpy==1.21.2`: xử lý dữ liệu dạng ma trận số
- `pandas==1.1.5`: xử lý dữ liệu dạng bảng
- `prettytable==2.4.0`: tiện ích in ra kết quả dạng bảng
- `scikit_learn==1.0.1`: xây dựng mô hình K-Nearest Neighbors
- `surprise==0.1`: thư viện build sẵn các thuật toán rating prediction sử dụng các phương pháp như SVD, SVD++, KNN,...

Để cài đặt các thư viện trên, thực hiện lệnh sau:
>`pip install -r requirements.txt`

## 3. Cách thực hiện demo
Tải repo này và truy cập vào thư mục chính:
> `git clone ...`

> `cd ...`
### Dự đoán rating
Để chạy thử nghiệm cho đề tài này, trước tiên cần tải file [checkpoint](#link-to-checkpoint) lưu trọng số mô hình mạng nơ-ron và đặt vào thư mục `model_checkpoint`.

Tiếp đến, tải dữ liệu [user genome](#link-to-user-genome) và [movie genome](#link-to-movie-genome) và đặt vào thư mục `data`.

Để thực hiện dự đoán đánh giá của người dùng `u` cho bộ phim `i`, thực thi file mã nguồn python `rating_predict` ở :
> `python src/rating_predict.py` 

Sau đó, danh sách các users sẽ được in ra, lưu ý rằng đây chỉ là id của 100 người dùng đầu tiên, bạn có thể thử với các id người dùng khác.

Nhập vào id của người dùng và id của bộ phim muốn dự đoán điểm đánh giá. Kết quả dự đoán sẽ được in ra terminal sau một thời gian thực thi. Kết quả thu được có dạng như sau:
```
Enter user id: 64
Enter movie id: 1000
-----------------------------------------
Movie name:  Curdled (1996)
Movie genre:  Crime
Actual rating:  None
Movie genome [[0.03475 0.0395  0.0145  ... 0.0075  0.09475 0.01825]]
User genome [[0.03924 0.03645 0.11525 ... 0.03097 0.07641 0.01837]]
-----------------------------------------
Predicted rating:  3.279511
```

### Đề xuất top K bộ phim

Để thực hiện đề xuất `k` bộ phim cho user có mã số `user id` bất kỳ, cần đảm bảo đã tải file checkpoint lưu vào thư mục `model_checkpoint` và file csv chứa dữ liệu genome của người dùng cũng như các bộ phim vào thư mục `data`.

Thực thi file mã nguồn python `topk_predict.py` bằng câu lệnh sau:
> `python src/topk_predict.py`

Nhập vào mã số của người dùng `user_id` và số lượng `k` bộ phim muốn đề xuất. Kết quả gợi ý được hiển thị có dạng như sau:
```
Enter user id: 20000
Enter K: 20
+---------------------------------------------+--------------------------+----------+------------------+--------------+
|                  Movie name                 |          Genre           | Distance | Estimated rating | Ground truth |
+---------------------------------------------+--------------------------+----------+------------------+--------------+
|      Pyromaniac's Love Story, A (1995)      |      Comedy|Romance      |  0.087   |      4.872       |     None     |
|               Babe, The (1992)              |          Drama           |  0.097   |      4.868       |     None     |
|           I'm Not Rappaport (1996)          |          Comedy          |  0.098   |      4.724       |     None     |
|           Robin and Marian (1976)           | Adventure|Drama|Romance  |  0.098   |       4.65       |     None     |
|         Month by the Lake, A (1995)         |   Comedy|Drama|Romance   |  0.099   |      4.649       |     None     |
|            Sandpiper, The (1965)            |      Drama|Romance       |  0.099   |      4.646       |     None     |
|                 Venus (2006)                |      Drama|Romance       |   0.1    |      4.591       |     None     |
|              Life Itself (2014)             |       Documentary        |   0.1    |       4.59       |     None     |
|             Natural, The (1984)             |          Drama           |  0.104   |      4.589       |     None     |
|        Only Angels Have Wings (1939)        | Adventure|Drama|Romance  |  0.105   |      4.528       |     None     |
|            Heaven Can Wait (1978)           |          Comedy          |  0.105   |      4.527       |     None     |
|         Same Time, Next Year (1978)         |   Comedy|Drama|Romance   |  0.108   |      4.497       |     None     |
|               Silverado (1985)              |      Action|Western      |  0.108   |      4.361       |     None     |
| Love Me If You Dare (Jeux d'enfants) (2003) |      Drama|Romance       |  0.109   |      4.352       |     None     |
|          Unfinished Life, An (2005)         |          Drama           |  0.109   |      4.298       |     None     |
|             Away from Her (2006)            |          Drama           |  0.109   |      4.298       |     None     |
|           Wedding Gift, The (1994)          |      Drama|Romance       |  0.109   |      4.247       |     None     |
|                 Wings (1927)                | Action|Drama|Romance|War |   0.11   |      4.246       |     None     |
|          Great Expectations (1946)          |          Drama           |   0.11   |       4.22       |     None     |
|                Ragtime (1981)               |          Drama           |   0.11   |      4.215       |     None     |
+---------------------------------------------+--------------------------+----------+------------------+--------------+
```

## 4. Đánh giá
- MAE: Đánh giá số tuyệt đối giữa giá trị rating dự đoán và rating thực tế.
- RMSE: Đánh giá căn bậc hai bình phương sai số giá trị rating dự đoán và rating thực tế. Hai thống số MAE và RMSE thu được qua quá trình huấn luyện sử dụng file `...`.

| Mô hình  | SVD-Surprise | KNN-Surprise | Mạng nơ-ron |
| ---------| -----------  | -----------  | ----------- |
| MAE      |  | | |
| RMSE     |  | | |

- Cummulative Hit Ratio: Tỉ lệ _hit_ của mô hình gợi ý, đánh giá chất lượng của mô hình KNN (k=20). Thông số này được tính ở file `...` sử dụng phương pháp Leave-one-out.

| Mô hình | Split 1  | Split 2  | Split 3  | Mean    |
| ------- | -------- | -------- | -------- |-------- | 
| CHR     | 0.024315 | 0.023758 |  |  |

## 5. Thành viên
- Trần Quang Duy
- Tôn Thất Hữu Trí 
- Đoàn Nguyễn Nam 
- Phan Thị Cẩm Tú
- Nguyễn Quang lực

## 6. Tài liệu tham khảo

1. a
2. 2
3. 3

