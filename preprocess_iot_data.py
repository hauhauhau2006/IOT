#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
preprocess_iot_data.py
------------------------
python preprocess_iot_data.py C:\Users\freed\IOT\ket_qua.csv --out ket_qua_preprocessed.csv --target Attack_type
"""

import argparse

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =========================================================================
# Cấu hình danh sách cột — CHỈNH Ở ĐÂY nếu nhóm muốn thay đổi chiến lược.
# Script chỉ áp dụng cho những cột thực sự có mặt trong tệp đầu vào, nên
# dùng được cả với CSV do extract_iot_pcap.py tạo ra (76 cột) lẫn CSV
# gốc kiểu Edge-IIoTset (61 đặc trưng + Attack_label/Attack_type).
# =========================================================================

# --- Cột định danh thô -> LOẠI BỎ hoàn toàn ---
# Lý do: đây là các giá trị gắn với MỘT thiết bị/phiên cụ thể trong lúc
# thu thập dữ liệu (địa chỉ IP/MAC, tên miền, URL, client id...). Nếu giữ
# lại, mô hình có nguy cơ ghi nhớ "IP này luôn là tấn công" thay vì học
# đặc trưng HÀNH VI mạng — không tổng quát hoá được sang mạng khác.
IDENTIFIER_COLUMNS = [
    'frame.time',
    'eth.src', 'eth.dst',
    'arp.src.hw_mac', 'arp.dst.hw_mac',
    'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'ip.src_host', 'ip.dst_host',
    'tcp.srcport', 'udp.srcport',        # cổng nguồn: ngẫu nhiên/ephemeral
    'http.request.uri', 'http.request.full_uri', 'http.host',
    'http.user_agent', 'http.response.phrase',
    'dns.qry.name', 'dns.a',
    'mqtt.clientid', 'mqtt.topic', 'mqtt.msg',
    'modbus.data',
]

# --- Cột phân loại (text/categorical) -> sẽ được MÃ HOÁ (không loại bỏ) ---
CATEGORICAL_COLUMNS = [
    'eth.type', 'ip.proto',
    'tcp.flags',
    'dns.qry.type', 'dns.flags.rcode',
    'http.request.method', 'http.response.code', 'http.content_type',
    'mqtt.msgtype_name', 'mqtt.hdrflags', 'mqtt.conflags',
    'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.msg_decoded_as',
    'modbus.func_name',
]

# --- Cột nhãn (label) — tách riêng khỏi tập đặc trưng X ---
LABEL_COLUMNS = ['Attack_label', 'Attack_type']


# =========================================================================
# Bước 1: Làm sạch dữ liệu
# =========================================================================

def clean_data(df, presence_check_cols=('ip.src_host', 'ip.dst_host', 'arp.opcode')):
    n0 = len(df)
    print(f'[Làm sạch] Số dòng ban đầu: {n0}')

    # (a) Chuẩn hoá ô rỗng '' / khoảng trắng thành NaN thực sự
    df = df.replace(r'^\s*$', pd.NA, regex=True)

    # (b) Loại bỏ dòng trùng lặp hoàn toàn (giống nhau ở TẤT CẢ các cột)
    dup_mask = df.duplicated()
    n_dup = int(dup_mask.sum())
    df = df[~dup_mask].copy()
    print(f'  - Dòng trùng lặp bị loại: {n_dup}')

    # (c) Loại bỏ dòng "lỗi": không xác định được cả tầng IP lẫn ARP
    #     (ví dụ: frame bị cắt cụt / giao thức lạ không dissect được)
    check_cols = [c for c in presence_check_cols if c in df.columns]
    if check_cols:
        malformed_mask = df[check_cols].isna().all(axis=1)
        n_malformed = int(malformed_mask.sum())
        df = df[~malformed_mask].copy()
    else:
        n_malformed = 0
    print(f'  - Dòng lỗi/không xác định giao thức bị loại: {n_malformed}')

    n1 = len(df)
    print(f'  - Số dòng sau làm sạch: {n1}  (loại tổng cộng {n0 - n1} dòng)')
    return df


def fill_missing(df, identifier_cols, categorical_cols, label_cols):
    """Điền giá trị thiếu: cột số -> 0, cột phân loại -> 'none'."""
    protected = set(identifier_cols) | set(categorical_cols) | set(label_cols)
    numeric_cols = [c for c in df.columns if c not in protected]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    n_numeric_na = int(df[numeric_cols].isna().sum().sum())
    df[numeric_cols] = df[numeric_cols].fillna(0)

    cat_present = [c for c in categorical_cols if c in df.columns]
    n_cat_na = int(df[cat_present].isna().sum().sum()) if cat_present else 0
    if cat_present:
        df[cat_present] = df[cat_present].fillna('none').astype(str)

    print(f'  - Ô NaN ở cột số đã điền 0: {n_numeric_na}')
    print(f'  - Ô NaN ở cột phân loại đã điền "none": {n_cat_na}')
    return df


# =========================================================================
# Bước 2: Loại bỏ đặc trưng định danh thừa
# =========================================================================

def drop_identifier_columns(df, identifier_cols):
    present = [c for c in identifier_cols if c in df.columns]
    print(f'[Loại định danh] Loại {len(present)} cột: {present}')
    return df.drop(columns=present)


def drop_near_constant_columns(df, exclude_cols, threshold=0.995):
    """Loại các cột mà >threshold tỉ lệ dòng có cùng 1 giá trị
    (gần như không đổi -> không mang thông tin phân biệt cho mô hình)."""
    to_drop = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        vc = df[c].value_counts(normalize=True, dropna=False)
        if len(vc) > 0 and vc.iloc[0] >= threshold:
            to_drop.append(c)
    print(f'[Giảm chiều] Loại {len(to_drop)} cột gần như không đổi '
          f'(>= {threshold:.1%} cùng 1 giá trị): {to_drop}')
    return df.drop(columns=to_drop)


# =========================================================================
# Bước 3: Mã hoá dữ liệu văn bản
# =========================================================================

def encode_categoricals(df, categorical_cols, onehot_max_unique=10):
    """Cardinality thấp -> One-Hot; cardinality cao hơn -> Label Encoding
    (tránh nổ số chiều nếu One-Hot toàn bộ)."""
    present = [c for c in categorical_cols if c in df.columns]
    onehot_cols, label_cols = [], []
    for c in present:
        n_unique = df[c].nunique(dropna=False)
        (onehot_cols if n_unique <= onehot_max_unique else label_cols).append(c)

    n_before = df.shape[1]
    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols)

    encoders = {}
    for c in label_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le

    print(f'[Mã hoá] One-Hot Encoding ({len(onehot_cols)} cột, '
          f'<= {onehot_max_unique} giá trị khác nhau): {onehot_cols}')
    print(f'[Mã hoá] Label Encoding ({len(label_cols)} cột, '
          f'> {onehot_max_unique} giá trị khác nhau): {label_cols}')
    print(f'  - Số cột trước mã hoá: {n_before} -> sau mã hoá: {df.shape[1]} '
          f'(One-Hot làm tăng số cột)')
    return df, encoders


def encode_label_columns(df, label_cols):
    """Mã hoá riêng cột nhãn (Attack_type dạng chuỗi -> số nguyên)."""
    encoders = {}
    for c in label_cols:
        if c not in df.columns:
            continue
        # Dùng is_numeric_dtype thay vì so sánh trực tiếp với `object`:
        # pandas >= 3.0 mặc định đọc cột chuỗi bằng kiểu StringDtype riêng
        # (không phải `object` truyền thống), so sánh == object sẽ bỏ sót.
        if not pd.api.types.is_numeric_dtype(df[c]):
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
            print(f'[Nhãn] Đã Label-Encode cột "{c}": '
                  f'{dict(zip(le.classes_, le.transform(le.classes_)))}')
    return df, encoders


# =========================================================================
# Pipeline tổng
# =========================================================================

def run_pipeline(df, target_col='Attack_type', onehot_max_unique=10,
                  drop_near_constant=True, near_constant_threshold=0.995):
    n_cols_raw = df.shape[1]
    n_rows_raw = df.shape[0]

    print('=' * 72)
    print(f'DỮ LIỆU ĐẦU VÀO: {n_rows_raw} dòng x {n_cols_raw} cột')
    print('=' * 72)

    # Bước 1: làm sạch
    df = clean_data(df)
    df = fill_missing(df, IDENTIFIER_COLUMNS, CATEGORICAL_COLUMNS, LABEL_COLUMNS)

    # Bước 2: loại đặc trưng định danh thừa
    print()
    df = drop_identifier_columns(df, IDENTIFIER_COLUMNS)
    n_cols_after_id = df.shape[1]

    protected_from_variance_check = set(CATEGORICAL_COLUMNS) | set(LABEL_COLUMNS)
    if drop_near_constant:
        df = drop_near_constant_columns(df, protected_from_variance_check,
                                         threshold=near_constant_threshold)
    n_cols_after_clean = df.shape[1]

    # Bước 3: mã hoá
    print()
    label_cols_present = [c for c in LABEL_COLUMNS if c in df.columns]
    df, label_encoders = encode_label_columns(df, label_cols_present)

    feature_cat_cols = [c for c in CATEGORICAL_COLUMNS if c not in LABEL_COLUMNS]
    df, cat_encoders = encode_categoricals(df, feature_cat_cols, onehot_max_unique)

    n_cols_final = df.shape[1]

    print()
    print('=' * 72)
    print('TÓM TẮT PIPELINE')
    print('=' * 72)
    print(f'Số dòng      : {n_rows_raw} -> {len(df)}')
    print(f'Số cột thô   : {n_cols_raw}')
    print(f'  Sau loại đặc trưng định danh/gần-hằng-số : {n_cols_after_clean} cột')
    print(f'  Sau mã hoá (One-Hot có thể làm tăng cột) : {n_cols_final} cột')
    print('=' * 72)

    return df, {'label_encoders': label_encoders, 'categorical_encoders': cat_encoders}


# =========================================================================
# Điểm vào chương trình
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline làm sạch + loại định danh + mã hoá dữ liệu đặc trưng IoT/IIoT.'
    )
    parser.add_argument('input_csv', help='Đường dẫn tệp CSV đầu vào')
    parser.add_argument('--out', default='du_lieu_da_xu_ly.csv',
                         help='Đường dẫn tệp CSV đầu ra (mặc định: du_lieu_da_xu_ly.csv)')
    parser.add_argument('--target', default='Attack_type',
                         help='Tên cột nhãn cần Label-Encode nếu có (mặc định: Attack_type)')
    parser.add_argument('--onehot-max-unique', type=int, default=10,
                         help='Ngưỡng số giá trị khác nhau để dùng One-Hot thay vì Label Encoding')
    parser.add_argument('--no-drop-near-constant', action='store_true',
                         help='Tắt bước loại cột gần như không đổi')
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)
    processed, _ = run_pipeline(
        df,
        target_col=args.target,
        onehot_max_unique=args.onehot_max_unique,
        drop_near_constant=not args.no_drop_near_constant,
    )
    processed.to_csv(args.out, index=False)
    print(f'\nĐã ghi dữ liệu đã xử lý -> {args.out}')


if __name__ == '__main__':
    main()