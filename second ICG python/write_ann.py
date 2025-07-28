import struct
import csv
from typing import List, Union
import numpy as np



def ann2int(ann_type: str) -> int:
    Typestr = 'NLRaVFJASEj/Q~|sT*D"=pB^t+u?![]en@xf(`)\'r'
    codeint = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22,
        23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 39,
        40, 40, 41
    ]
    idx = Typestr.find(ann_type)
    return codeint[idx] if idx != -1 else 0


def write_ann(record_name: str,
              HRVparams: dict,
              annotator: str,
              ann: List[int],
              ann_type: Union[List[str], str] = 'N',
              sub_type: Union[List[int], int] = 0,
              chan: Union[List[int], int] = 0,
              num: Union[List[int], int] = 0,
              comments: Union[List[str], str] = ''):

    # Ensure all inputs are lists of proper length
    N = len(ann)
    if isinstance(ann_type, str):
        ann_type = [ann_type] * N
    if isinstance(sub_type, int):
        sub_type = [sub_type] * N
    if isinstance(chan, int):
        chan = [chan] * N
    if isinstance(num, int):
        num = [num] * N
    if isinstance(comments, str):
        comments = [''] * N

    if HRVparams['output']['ann_format'] == 'binary':
        byte_write = bytearray()
        annfile = f"{record_name}.{annotator}"
        ann_pre = 0

        for i in range(N):
            # 检查 ann[i] 的类型，避免 list 与 int 相减报错
            if isinstance(ann[i], (list, np.ndarray)):
                if len(ann[i]) == 0:
                    print(f"Warning: ann[{i}] is empty, skipping this annotation")
                    continue
                anntime = np.array(ann[i]).flatten()[0] - ann_pre
            else:
                anntime = ann[i] - ann_pre

            typei = ann2int(ann_type[i])
            if anntime <= 1023:
                byte1 = anntime & 0xFF
                byte2 = ((anntime >> 8) & 0x03) | (typei << 2)
                byte_write.extend([byte1, byte2])
            else:
                # long annotation
                byte_write.extend([0, 59 << 2])
                anntime_L = ann[i] - ann_pre
                byte_write.extend([
                    (anntime_L >> 16) & 0xFF,
                    (anntime_L >> 24) & 0xFF,
                    anntime_L & 0xFF,
                    (anntime_L >> 8) & 0xFF
                ])
                byte_write.extend([0, typei << 2])

            # subtype
            if sub_type[i] != 0:
                byte_write.extend([
                    sub_type[i] & 0xFF,
                    ((sub_type[i] >> 8) & 0x03) | (61 << 2)
                ])

            # first annotation only
            if i == 0:
                if chan[i] != 0:
                    byte_write.extend([
                        chan[i] & 0xFF,
                        ((chan[i] >> 8) & 0x03) | (62 << 2)
                    ])
                if num[i] != 0:
                    byte_write.extend([
                        num[i] & 0xFF,
                        ((num[i] >> 8) & 0x03) | (60 << 2)
                    ])
            else:
                if chan[i] != chan[i - 1]:
                    byte_write.extend([
                        chan[i] & 0xFF,
                        ((chan[i] >> 8) & 0x03) | (62 << 2)
                    ])
                if num[i] != num[i - 1]:
                    byte_write.extend([
                        num[i] & 0xFF,
                        ((num[i] >> 8) & 0x03) | (60 << 2)
                    ])

            # comments
            if comments[i]:
                com_bytes = comments[i].encode('ascii')
                com_len = len(com_bytes)
                byte_write.extend([com_len & 0xFF, ((com_len >> 8) & 0x03) | (63 << 2)])
                byte_write.extend(com_bytes)
                if com_len % 2 == 1:
                    byte_write.append(0)

            ann_pre = ann[i]

        byte_write.extend([0, 0])
        with open(annfile, 'wb') as f:
            f.write(byte_write)

    elif HRVparams['output']['ann_format'] == 'csv':
        filename = f"{record_name}.{annotator}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ann', 'annType', 'subType', 'chan', 'num', 'comments'])
            for i in range(N):
                writer.writerow([
                    ann[i],
                    ann_type[i],
                    sub_type[i],
                    chan[i],
                    num[i],
                    comments[i]
                ])
