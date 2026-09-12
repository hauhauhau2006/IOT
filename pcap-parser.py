#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tên thư mục: pcap-parser.py
--------------------
    python pcap-parser.py [link] --out ket_qua.csv --top 15
"""
import argparse
import csv
import socket
import struct
from collections import Counter
from datetime import datetime, timezone

import dpkt


# =========================================================================
# Hàm tiện ích dùng chung
# =========================================================================

def mac_to_str(mac_bytes):
    return ':'.join('%02x' % b for b in mac_bytes)


def ip_to_str(ip_bytes):
    try:
        return socket.inet_ntoa(ip_bytes)
    except (OSError, socket.error):
        return socket.inet_ntop(socket.AF_INET6, ip_bytes)


def ts_to_iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


_TCP_FLAG_BITS = [
    ('FIN', dpkt.tcp.TH_FIN), ('SYN', dpkt.tcp.TH_SYN),
    ('RST', dpkt.tcp.TH_RST), ('PSH', dpkt.tcp.TH_PUSH),
    ('ACK', dpkt.tcp.TH_ACK), ('URG', dpkt.tcp.TH_URG),
]


def tcp_flags_to_str(flags):
    names = [n for n, bit in _TCP_FLAG_BITS if flags & bit]
    return '|'.join(names) if names else '-'


_IP_PROTO_NAMES = {1: 'ICMP', 6: 'TCP', 17: 'UDP', 2: 'IGMP', 58: 'ICMPv6'}

_DNS_TYPE_NAMES = {
    dpkt.dns.DNS_A: 'A', dpkt.dns.DNS_NS: 'NS', dpkt.dns.DNS_CNAME: 'CNAME',
    dpkt.dns.DNS_SOA: 'SOA', dpkt.dns.DNS_PTR: 'PTR', dpkt.dns.DNS_MX: 'MX',
    dpkt.dns.DNS_TXT: 'TXT', dpkt.dns.DNS_AAAA: 'AAAA', dpkt.dns.DNS_SRV: 'SRV',
}


def dns_type_name(t):
    return _DNS_TYPE_NAMES.get(t, str(t))


# =========================================================================
# Bộ giải mã MQTT (dpkt không hỗ trợ sẵn) — MQTT v3.1.1
# =========================================================================

MQTT_TYPES = {
    1: 'CONNECT', 2: 'CONNACK', 3: 'PUBLISH', 4: 'PUBACK', 5: 'PUBREC',
    6: 'PUBREL', 7: 'PUBCOMP', 8: 'SUBSCRIBE', 9: 'SUBACK',
    10: 'UNSUBSCRIBE', 11: 'UNSUBACK', 12: 'PINGREQ', 13: 'PINGRESP',
    14: 'DISCONNECT', 15: 'AUTH',
}

# Cổng TCP mặc định của MQTT / MQTT-over-TLS
MQTT_PORTS = (1883, 8883)
MODBUS_PORT = 502


def _decode_varint_remaining_length(data, pos):
    """Giải mã trường 'Remaining Length' dạng variable-length của MQTT."""
    multiplier = 1
    value = 0
    while True:
        if pos >= len(data):
            raise ValueError('thiếu dữ liệu khi đọc remaining length')
        b = data[pos]
        pos += 1
        value += (b & 0x7F) * multiplier
        if (b & 0x80) == 0:
            break
        multiplier *= 128
        if multiplier > 128 ** 3:
            raise ValueError('remaining length không hợp lệ')
    return value, pos


def _read_mqtt_utf8_str(data, off):
    """Đọc 1 chuỗi UTF-8 dạng 'độ dài 2 byte + nội dung' theo chuẩn MQTT."""
    if off + 2 > len(data):
        return None, off
    (length,) = struct.unpack('>H', data[off:off + 2])
    off += 2
    raw = data[off:off + length]
    off += len(raw)  # nếu gói bị cắt bớt (snaplen) thì lấy phần còn lại
    try:
        return raw.decode('utf-8', errors='replace'), off
    except Exception:
        return raw.hex(), off


def _decode_mqtt_payload(payload, max_len=200):
    """Trả về (nội dung, cách_giải_mã) — giống cột mqtt.msg_decoded_as của Wireshark."""
    if not payload:
        return '', 'empty'
    try:
        s = payload.decode('utf-8')
        if all(32 <= ord(c) < 127 or c in '\r\n\t' for c in s):
            return s[:max_len], 'text'
    except Exception:
        pass
    return payload[:max_len].hex(), 'hex'


def parse_mqtt(data):
    """Phân tích MỘT gói tin MQTT nằm ở đầu payload TCP. Trả về dict cột hoặc None."""
    if len(data) < 2:
        return None
    b0 = data[0]
    msgtype = (b0 >> 4) & 0x0F
    flags = b0 & 0x0F
    if msgtype not in MQTT_TYPES:
        return None
    try:
        rem_len, pos = _decode_varint_remaining_length(data, 1)
    except ValueError:
        return None

    body = data[pos:pos + rem_len]
    out = {
        'mqtt.msgtype': msgtype,
        'mqtt.msgtype_name': MQTT_TYPES[msgtype],
        'mqtt.hdrflags': hex(flags),
        'mqtt.len': rem_len,
    }

    try:
        if msgtype == 1:  # CONNECT
            off = 0
            proto_name, off = _read_mqtt_utf8_str(body, off)
            out['mqtt.protoname'] = proto_name
            if off < len(body):
                out['mqtt.ver'] = body[off]; off += 1
            if off < len(body):
                cflags = body[off]; off += 1
                out['mqtt.conflags'] = hex(cflags)
                out['mqtt.conflag.cleansess'] = (cflags >> 1) & 0x1
            if off + 2 <= len(body):
                out['mqtt.kalive'] = struct.unpack('>H', body[off:off + 2])[0]
                off += 2
            client_id, off = _read_mqtt_utf8_str(body, off)
            out['mqtt.clientid'] = client_id

        elif msgtype == 2:  # CONNACK
            if len(body) >= 2:
                out['mqtt.conack.flags'] = hex(body[0])
                out['mqtt.conack.val'] = body[1]

        elif msgtype == 3:  # PUBLISH
            qos = (flags >> 1) & 0x3
            out['mqtt.dupflag'] = (flags >> 3) & 0x1
            out['mqtt.qos'] = qos
            out['mqtt.retain'] = flags & 0x1
            off = 0
            topic, off = _read_mqtt_utf8_str(body, off)
            out['mqtt.topic'] = topic
            out['mqtt.topic_len'] = len(topic) if topic else 0
            if qos > 0 and off + 2 <= len(body):
                off += 2  # Packet Identifier — không cần cho thống kê này
            msg, decoded_as = _decode_mqtt_payload(body[off:])
            out['mqtt.msg'] = msg
            out['mqtt.msg_decoded_as'] = decoded_as

        elif msgtype == 8:  # SUBSCRIBE
            off = 2  # 2 byte đầu là Packet Identifier
            topics = []
            while off < len(body):
                t, off = _read_mqtt_utf8_str(body, off)
                if t is None:
                    break
                topics.append(t)
                off += 1  # 1 byte QoS yêu cầu, theo sau mỗi topic filter
            out['mqtt.topic'] = ';'.join(topics)
            out['mqtt.topic_len'] = sum(len(t) for t in topics)
    except (struct.error, IndexError):
        # Payload bị cắt giữa chừng (do snaplen) — giữ lại các trường đã đọc được
        pass

    return out


# =========================================================================
# Bộ giải mã Modbus/TCP (dpkt không hỗ trợ sẵn) — MBAP header + PDU
# =========================================================================

MODBUS_FUNC_NAMES = {
    1: 'Read Coils', 2: 'Read Discrete Inputs', 3: 'Read Holding Registers',
    4: 'Read Input Registers', 5: 'Write Single Coil', 6: 'Write Single Register',
    7: 'Read Exception Status', 8: 'Diagnostics', 11: 'Get Comm Event Counter',
    15: 'Write Multiple Coils', 16: 'Write Multiple Registers',
    17: 'Report Slave ID', 22: 'Mask Write Register',
    23: 'Read/Write Multiple Registers', 43: 'Encapsulated Interface Transport',
}


def parse_modbus(data):
    """Phân tích MBAP header + function code của Modbus/TCP."""
    if len(data) < 8:
        return None
    trans_id, proto_id, length, unit_id = struct.unpack('>HHHB', data[:7])
    if proto_id != 0:
        return None  # proto_id phải luôn = 0 với Modbus/TCP hợp lệ
    func_code = data[7]
    is_exception = bool(func_code & 0x80)
    base_func = func_code & 0x7F
    return {
        'mbtcp.trans_id': trans_id,
        'mbtcp.proto_id': proto_id,
        'mbtcp.len': length,
        'mbtcp.unit_id': unit_id,
        'modbus.func_code': func_code,
        'modbus.func_name': ('Exception:' if is_exception else '')
        + MODBUS_FUNC_NAMES.get(base_func, str(base_func)),
        'modbus.data': data[8:40].hex(),  # cắt bớt để cột không quá dài
    }


# =========================================================================
# Sơ đồ cột đầu ra (bảng CSV phẳng — 1 dòng / 1 gói tin)
# =========================================================================

FIELDNAMES = [
    # --- Frame / Ethernet ---
    'frame.time', 'frame.len', 'eth.src', 'eth.dst', 'eth.type',
    # --- ARP (quan trọng cho phát hiện ARP Spoofing / MITM) ---
    'arp.opcode', 'arp.hw.size',
    'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'arp.src.hw_mac', 'arp.dst.hw_mac',
    # --- IP ---
    'ip.src_host', 'ip.dst_host', 'ip.proto', 'ip.ttl', 'ip.len',
    # --- ICMP ---
    'icmp.type', 'icmp.code', 'icmp.checksum', 'icmp.seq_le',
    # --- TCP ---
    'tcp.srcport', 'tcp.dstport', 'tcp.seq', 'tcp.ack', 'tcp.len',
    'tcp.flags', 'tcp.flags.syn', 'tcp.flags.ack', 'tcp.flags.fin',
    'tcp.flags.rst', 'tcp.window_size',
    # --- UDP ---
    'udp.srcport', 'udp.dstport', 'udp.length',
    # --- DNS ---
    'dns.qry.name', 'dns.qry.name.len', 'dns.qry.type', 'dns.qr',
    'dns.flags.rcode', 'dns.retransmission', 'dns.a',
    # --- HTTP ---
    'http.request.method', 'http.request.uri', 'http.request.version',
    'http.request.full_uri', 'http.host', 'http.user_agent',
    'http.response.code', 'http.response.phrase',
    'http.content_type', 'http.content_length',
    # --- MQTT (IIoT) ---
    'mqtt.msgtype', 'mqtt.msgtype_name', 'mqtt.hdrflags', 'mqtt.len',
    'mqtt.dupflag', 'mqtt.qos', 'mqtt.retain',
    'mqtt.protoname', 'mqtt.ver', 'mqtt.conflags', 'mqtt.conflag.cleansess',
    'mqtt.kalive', 'mqtt.clientid', 'mqtt.conack.flags',
    'mqtt.topic', 'mqtt.topic_len', 'mqtt.msg', 'mqtt.msg_decoded_as',
    # --- Modbus/TCP (IIoT) ---
    'mbtcp.trans_id', 'mbtcp.proto_id', 'mbtcp.len', 'mbtcp.unit_id',
    'modbus.func_code', 'modbus.func_name', 'modbus.data',
]


def empty_row():
    return dict.fromkeys(FIELDNAMES, '')


# =========================================================================
# Bộ phân tích chính
# =========================================================================

class PcapAnalyzer:

    def __init__(self, pcap_path):
        self.pcap_path = pcap_path
        self.rows = []  # danh sách dict, mỗi phần tử = 1 dòng CSV phẳng

        # Thống kê tổng hợp cho báo cáo tóm tắt
        self.total_packets = 0
        self.total_bytes = 0
        self.malformed_packets = 0
        self.other_eth_packets = 0  # không phải ARP/IP/IPv6 (vd. VLAN tag lạ...)
        self.start_ts = None
        self.end_ts = None

        self.eth_addr_counter = Counter()
        self.ip_src_counter = Counter()
        self.ip_dst_counter = Counter()
        self.ip_proto_counter = Counter()
        self.tcp_port_counter = Counter()
        self.udp_port_counter = Counter()
        self.conversations = Counter()

        self.arp_opcode_counter = Counter()
        self.icmp_type_counter = Counter()
        self.dns_query_counter = Counter()
        self.http_host_counter = Counter()
        self.mqtt_msgtype_counter = Counter()
        self.mqtt_topic_counter = Counter()
        self.modbus_func_counter = Counter()

        # Dùng để nhận diện DNS retransmission (truy vấn lặp lại do timeout)
        self._dns_seen = Counter()

    # ---- vòng lặp chính -----------------------------------------------
    def run(self):
        with open(self.pcap_path, 'rb') as f:
            for ts, buf in self._open_reader(f):
                self.total_packets += 1
                self.total_bytes += len(buf)
                self.start_ts = ts if self.start_ts is None else min(self.start_ts, ts)
                self.end_ts = ts if self.end_ts is None else max(self.end_ts, ts)
                self._process_packet(ts, buf)
        return self

    @staticmethod
    def _open_reader(f):
        try:
            return dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            return dpkt.pcapng.Reader(f)

    # ---- xử lý 1 gói tin -> 1 dòng CSV phẳng --------------------------
    def _process_packet(self, ts, buf):
        row = empty_row()
        row['frame.time'] = ts_to_iso(ts)
        row['frame.len'] = len(buf)

        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception:
            self.malformed_packets += 1
            return

        eth_src, eth_dst = mac_to_str(eth.src), mac_to_str(eth.dst)
        row['eth.src'], row['eth.dst'], row['eth.type'] = eth_src, eth_dst, hex(eth.type)
        self.eth_addr_counter[eth_src] += 1

        # --- ARP: chạy thẳng trên Ethernet, không có tầng IP ở trên ---
        if eth.type == dpkt.ethernet.ETH_TYPE_ARP and isinstance(eth.data, dpkt.arp.ARP):
            self._fill_arp(row, eth.data)
            self.rows.append(row)
            return

        ip = eth.data
        is_ipv4 = isinstance(ip, dpkt.ip.IP)
        is_ipv6 = isinstance(ip, dpkt.ip6.IP6)
        if not (is_ipv4 or is_ipv6):
            self.other_eth_packets += 1
            self.rows.append(row)
            return

        ip_src, ip_dst = ip_to_str(ip.src), ip_to_str(ip.dst)
        proto_num = ip.p if is_ipv4 else ip.nxt
        proto_name = _IP_PROTO_NAMES.get(proto_num, str(proto_num))

        self.ip_src_counter[ip_src] += 1
        self.ip_dst_counter[ip_dst] += 1
        self.ip_proto_counter[proto_name] += 1
        self.conversations[(ip_src, ip_dst)] += 1

        row.update({
            'ip.src_host': ip_src, 'ip.dst_host': ip_dst,
            'ip.proto': proto_name,
            'ip.ttl': getattr(ip, 'ttl', getattr(ip, 'hlim', '')),
            'ip.len': getattr(ip, 'len', ''),
        })

        transport = ip.data

        if isinstance(transport, dpkt.icmp.ICMP):
            self._fill_icmp(row, transport)

        elif isinstance(transport, dpkt.tcp.TCP):
            tcp = transport
            self.tcp_port_counter[tcp.sport] += 1
            self.tcp_port_counter[tcp.dport] += 1
            row.update({
                'tcp.srcport': tcp.sport, 'tcp.dstport': tcp.dport,
                'tcp.seq': tcp.seq, 'tcp.ack': tcp.ack,
                'tcp.len': len(tcp.data), 'tcp.flags': hex(tcp.flags),
                'tcp.flags.syn': int(bool(tcp.flags & dpkt.tcp.TH_SYN)),
                'tcp.flags.ack': int(bool(tcp.flags & dpkt.tcp.TH_ACK)),
                'tcp.flags.fin': int(bool(tcp.flags & dpkt.tcp.TH_FIN)),
                'tcp.flags.rst': int(bool(tcp.flags & dpkt.tcp.TH_RST)),
                'tcp.window_size': tcp.win,
            })
            if tcp.data:
                if tcp.sport == 80 or tcp.dport == 80:
                    self._fill_http(row, tcp.data)
                elif tcp.sport in MQTT_PORTS or tcp.dport in MQTT_PORTS:
                    self._fill_mqtt(row, tcp.data)
                elif tcp.sport == MODBUS_PORT or tcp.dport == MODBUS_PORT:
                    self._fill_modbus(row, tcp.data)

        elif isinstance(transport, dpkt.udp.UDP):
            udp = transport
            self.udp_port_counter[udp.sport] += 1
            self.udp_port_counter[udp.dport] += 1
            row.update({
                'udp.srcport': udp.sport, 'udp.dstport': udp.dport,
                'udp.length': len(udp.data),
            })
            if udp.sport == 53 or udp.dport == 53:
                self._fill_dns(row, ip_src, ip_dst, udp.data)

        self.rows.append(row)

    # ---- ARP -------------------------------------------------------------
    def _fill_arp(self, row, arp):
        self.arp_opcode_counter[arp.op] += 1
        row['arp.opcode'] = arp.op
        row['arp.hw.size'] = arp.hln
        row['arp.src.hw_mac'] = mac_to_str(arp.sha)
        row['arp.dst.hw_mac'] = mac_to_str(arp.tha)
        try:
            row['arp.src.proto_ipv4'] = ip_to_str(arp.spa)
            row['arp.dst.proto_ipv4'] = ip_to_str(arp.tpa)
        except Exception:
            pass

    # ---- ICMP --------------------------------------------------------------
    def _fill_icmp(self, row, icmp):
        self.icmp_type_counter[icmp.type] += 1
        row['icmp.type'] = icmp.type
        row['icmp.code'] = icmp.code
        row['icmp.checksum'] = icmp.sum
        seq = getattr(icmp.data, 'seq', '')
        row['icmp.seq_le'] = seq

    # ---- HTTP --------------------------------------------------------------
    def _fill_http(self, row, payload):
        try:
            if payload.startswith((b'GET ', b'POST', b'PUT ', b'DELETE',
                                    b'HEAD', b'OPTIONS', b'PATCH', b'CONNECT')):
                req = dpkt.http.Request(payload)
                host = req.headers.get('host', '')
                row.update({
                    'http.request.method': req.method,
                    'http.request.uri': req.uri,
                    'http.request.version': req.version,
                    'http.request.full_uri': (host + req.uri) if host else req.uri,
                    'http.host': host,
                    'http.user_agent': req.headers.get('user-agent', ''),
                })
                if host:
                    self.http_host_counter[host] += 1
            elif payload.startswith(b'HTTP/'):
                resp = dpkt.http.Response(payload)
                row.update({
                    'http.response.code': resp.status,
                    'http.response.phrase': resp.reason,
                    'http.content_type': resp.headers.get('content-type', ''),
                    'http.content_length': resp.headers.get('content-length', ''),
                })
        except Exception:
            pass  # payload không phải HTTP hoàn chỉnh (bị cắt/giao thức khác)

    # ---- MQTT --------------------------------------------------------------
    def _fill_mqtt(self, row, payload):
        info = parse_mqtt(payload)
        if not info:
            return
        row.update({k: v for k, v in info.items() if v is not None})
        self.mqtt_msgtype_counter[info.get('mqtt.msgtype_name', '?')] += 1
        if info.get('mqtt.topic'):
            self.mqtt_topic_counter[info['mqtt.topic']] += 1

    # ---- Modbus/TCP --------------------------------------------------------
    def _fill_modbus(self, row, payload):
        info = parse_modbus(payload)
        if not info:
            return
        row.update(info)
        self.modbus_func_counter[info['modbus.func_name']] += 1

    # ---- DNS -----------------------------------------------------------
    def _fill_dns(self, row, ip_src, ip_dst, data):
        if not data:
            return
        try:
            dns = dpkt.dns.DNS(data)
        except Exception:
            return

        is_response = (dns.qr == dpkt.dns.DNS_R)
        query_name = dns.qd[0].name if dns.qd else ''
        query_type = dns_type_name(dns.qd[0].type) if dns.qd else ''

        row.update({
            'dns.qry.name': query_name,
            'dns.qry.name.len': len(query_name),
            'dns.qry.type': query_type,
            'dns.qr': int(is_response),
            'dns.flags.rcode': dns.rcode,
        })

        if query_name:
            self.dns_query_counter[query_name] += 1
            # Nhận diện truy vấn lặp lại (retransmission) theo txn id + tên miền
            key = (ip_src, ip_dst, dns.id, query_name)
            row['dns.retransmission'] = int(self._dns_seen[key] > 0)
            self._dns_seen[key] += 1

        if is_response and dns.an:
            for rr in dns.an:
                if rr.type == dpkt.dns.DNS_A:
                    row['dns.a'] = ip_to_str(rr.rdata)
                    break

    # ---- báo cáo tóm tắt ------------------------------------------------
    def print_summary(self, top_n=10):
        duration = (self.end_ts - self.start_ts) if (self.start_ts and self.end_ts) else 0
        print('=' * 72)
        print(f'TỆP PHÂN TÍCH : {self.pcap_path}')
        print('=' * 72)
        print(f'Tổng số gói tin        : {self.total_packets}')
        print(f'Tổng dung lượng        : {self.total_bytes:,} bytes')
        print(f'Thời gian bắt gói      : {duration:.2f} giây')
        print(f'Gói tin lỗi/không đọc  : {self.malformed_packets}')
        print(f'Gói Ethernet khác (không ARP/IP) : {self.other_eth_packets}')

        print('\n--- Phân bố giao thức tầng IP ---')
        for p, c in self.ip_proto_counter.most_common():
            print(f'  {p:<8}: {c}')

        if self.arp_opcode_counter:
            print('\n--- ARP ---')
            names = {1: 'Request', 2: 'Reply'}
            for op, c in self.arp_opcode_counter.most_common():
                print(f'  opcode {op} ({names.get(op, "?"):<8}) : {c} gói')

        if self.icmp_type_counter:
            print('\n--- ICMP (theo type) ---')
            for t, c in self.icmp_type_counter.most_common():
                print(f'  type {t:<3}: {c} gói')

        print(f'\n--- Top {top_n} địa chỉ IP nguồn ---')
        for a, c in self.ip_src_counter.most_common(top_n):
            print(f'  {a:<20} {c} gói')

        print(f'\n--- Top {top_n} cặp giao tiếp (IP nguồn -> IP đích) ---')
        for (s, d), c in self.conversations.most_common(top_n):
            print(f'  {s:<15} -> {d:<15}  {c} gói')

        print(f'\n--- Top {top_n} cổng TCP ---')
        for p, c in self.tcp_port_counter.most_common(top_n):
            print(f'  cổng {p:<6} {c} gói')

        print(f'\n--- Top {top_n} cổng UDP ---')
        for p, c in self.udp_port_counter.most_common(top_n):
            print(f'  cổng {p:<6} {c} gói')

        if self.dns_query_counter:
            print(f'\n--- Top {top_n} tên miền được tra cứu (DNS) ---')
            for name, c in self.dns_query_counter.most_common(top_n):
                print(f'  {name:<40} {c} lần')

        if self.http_host_counter:
            print(f'\n--- Top {top_n} Host được truy cập qua HTTP ---')
            for host, c in self.http_host_counter.most_common(top_n):
                print(f'  {host:<40} {c} yêu cầu')

        if self.mqtt_msgtype_counter:
            print('\n--- MQTT: phân bố theo loại gói tin ---')
            for t, c in self.mqtt_msgtype_counter.most_common():
                print(f'  {t:<12}: {c}')
            if self.mqtt_topic_counter:
                print(f'--- Top {top_n} MQTT topic ---')
                for topic, c in self.mqtt_topic_counter.most_common(top_n):
                    print(f'  {topic:<40} {c} gói')

        if self.modbus_func_counter:
            print('\n--- Modbus/TCP: phân bố theo function code ---')
            for name, c in self.modbus_func_counter.most_common():
                print(f'  {name:<35}: {c}')

        print('=' * 72)

    # ---- xuất CSV phẳng duy nhất -----------------------------------------
    def export_csv(self, out_path):
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f'Đã ghi {len(self.rows)} dòng (1 dòng/gói tin) -> {out_path}')


# =========================================================================
# Điểm vào chương trình
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=('Trích xuất đặc trưng Ethernet/ARP/IP/ICMP/TCP/UDP/'
                      'DNS/HTTP/MQTT/Modbus từ tệp .pcap bằng dpkt, '
                      'xuất ra 1 bảng CSV phẳng (1 dòng = 1 gói tin).')
    )
    parser.add_argument('pcap_file', help='Đường dẫn tới tệp .pcap cần phân tích')
    parser.add_argument('--out', default='iot_packets_flat.csv',
                         help='Đường dẫn tệp CSV đầu ra (mặc định: iot_packets_flat.csv)')
    parser.add_argument('--top', type=int, default=10,
                         help='Số lượng mục hiển thị trong mỗi bảng xếp hạng (mặc định: 10)')
    args = parser.parse_args()

    analyzer = PcapAnalyzer(args.pcap_file).run()
    analyzer.print_summary(top_n=args.top)
    analyzer.export_csv(args.out)


if __name__ == '__main__':
    main()