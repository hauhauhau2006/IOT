#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chi_square.py
-------------------------
Chọn các đặc trưng tốt nhất bằng kiểm định Chi-square (Chi-square test of
independence), áp dụng trên dữ liệu ĐÃ tiền xử lý (đầu ra của
preprocess_iot_data.py — toàn bộ cột đã ở dạng số). Mục tiêu: giảm số
chiều để giảm chi phí tính toán/bộ nhớ khi triển khai mô hình trên
thiết bị IoT.

LƯU Ý QUAN TRỌNG: Chi-square là phương pháp chọn đặc trưng CÓ GIÁM SÁT
(supervised feature selection) — nó đo mức độ PHỤ THUỘC THỐNG KÊ giữa
từng đặc trưng và NHÃN LỚP (Attack_type / Attack_label). Vì vậy:
    - Dữ liệu đầu vào BẮT BUỘC phải có cột nhãn.
    - Nếu không có (ví dụ CSV được trích xuất thẳng từ pcap chưa gán
      nhãn), script sẽ báo lỗi rõ ràng thay vì tự suy đoán/gán nhãn giả —
      chọn đặc trưng dựa trên nhãn sai sẽ cho ra kết quả vô nghĩa.

Cách dùng:
    python Chi_square.py ket_qua_preprocessed.csv --target Attack_type --k 60
"""

import argparse

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def resolve_target_column(df, requested_target):
    """Nếu cột nhãn yêu cầu không tồn tại, thử phương án dự phòng
    (Attack_type <-> Attack_label). Nếu cả hai đều không có -> báo lỗi."""
    if requested_target in df.columns:
        return requested_target
    fallback = 'Attack_label' if requested_target == 'Attack_type' else 'Attack_type'
    if fallback in df.columns:
        print(f'[Cảnh báo] Không thấy cột "{requested_target}", '
              f'dùng cột nhãn thay thế: "{fallback}"')
        return fallback
    raise SystemExit(
        f'LỖI: Không tìm thấy cột nhãn nào ("{requested_target}" hoặc '
        f'"{fallback}") trong dữ liệu đầu vào.\n'
        f'Các cột hiện có: {list(df.columns)}\n\n'
        'Kiểm định Chi-square là phương pháp CÓ GIÁM SÁT (supervised) — '
        'bắt buộc phải có nhãn lớp (Attack_type/Attack_label) để đo mức độ '
        'phụ thuộc giữa từng đặc trưng và nhãn. Nếu tệp của bạn được trích '
        'xuất trực tiếp từ pcap và chưa gắn nhãn, hãy gán nhãn cho từng '
        'dòng (theo tệp pcap nguồn — ví dụ toàn bộ dòng từ '
        'Backdoor_attack.pcap -> Attack_type="Backdoor") TRƯỚC khi chạy '
        'bước chọn đặc trưng này.'
    )


def ensure_nonnegative(X):
    """chi2 trong sklearn yêu cầu mọi giá trị đặc trưng >= 0. Nếu có cột
    âm, scale cột đó về [0, 1] bằng MinMaxScaler thay vì loại bỏ, để
    không mất thông tin."""
    X = X.copy()
    neg_cols = [c for c in X.columns if (X[c] < 0).any()]
    if neg_cols:
        print(f'[Cảnh báo] {len(neg_cols)} cột có giá trị âm (không hợp lệ '
              f'với chi2), tự động scale về [0, 1]: {neg_cols}')
        scaler = MinMaxScaler()
        X[neg_cols] = scaler.fit_transform(X[neg_cols])
    return X


def select_features(X, y, k):
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)

    scores = pd.DataFrame({
        'feature': X.columns,
        'chi2_score': selector.scores_,
        'p_value': selector.pvalues_,
    }).sort_values('chi2_score', ascending=False).reset_index(drop=True)

    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features, scores


def run(input_csv, target_col, k, out_path):
    df = pd.read_csv(input_csv, low_memory=False)
    n_rows, n_cols_in = df.shape
    print(f'Dữ liệu đầu vào: {n_rows} dòng x {n_cols_in} cột')

    target_col = resolve_target_column(df, target_col)

    y_raw = df[target_col]
    if not pd.api.types.is_numeric_dtype(y_raw):
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        y = y_raw.values

    label_cols = [c for c in ['Attack_label', 'Attack_type'] if c in df.columns]
    X = df.drop(columns=label_cols)
    n_features_available = X.shape[1]

    X = ensure_nonnegative(X)
    selected_features, scores = select_features(X, y, k)

    print()
    print('=' * 72)
    print(f'BẢNG XẾP HẠNG CHI-SQUARE (toàn bộ {len(scores)} đặc trưng, sắp xếp giảm dần)')
    print('=' * 72)
    with pd.option_context('display.max_rows', None, 'display.width', 100):
        print(scores.to_string(index=False))

    print()
    print(f'-> Giữ lại TOP {len(selected_features)}/{n_features_available} đặc trưng tốt nhất:')
    for f in selected_features:
        row = scores[scores['feature'] == f].iloc[0]
        print(f'   - {f:<35} chi2={row.chi2_score:.2f}  p={row.p_value:.2e}')

    out_df = pd.concat([df[selected_features], df[label_cols]], axis=1)
    out_df.to_csv(out_path, index=False)

    print()
    print('=' * 72)
    print('TÓM TẮT')
    print('=' * 72)
    print(f'Nhãn dùng để kiểm định : {target_col}')
    print(f'Số đặc trưng           : {n_features_available} -> {len(selected_features)}')
    print(f'Đã ghi -> {out_path} ({out_df.shape[0]} dòng x {out_df.shape[1]} cột, '
          f'gồm {len(selected_features)} đặc trưng + {len(label_cols)} cột nhãn)')

    return out_df, scores


def main():
    parser = argparse.ArgumentParser(
        description='Chọn đặc trưng tốt nhất bằng kiểm định Chi-square (SelectKBest + chi2).'
    )
    parser.add_argument('input_csv', help='Tệp CSV đã qua tiền xử lý (toàn bộ cột dạng số)')
    parser.add_argument('--target', default='Attack_type',
                         help='Tên cột nhãn (mặc định: Attack_type, tự thử Attack_label nếu không có)')
    parser.add_argument('--k', type=int, default=60,
                         help='Số đặc trưng muốn giữ lại (mặc định: 60)')
    parser.add_argument('--out', default='du_lieu_da_chon_dac_trung.csv',
                         help='Đường dẫn CSV đầu ra')
    args = parser.parse_args()
    run(args.input_csv, args.target, args.k, args.out)


if __name__ == '__main__':
    main()