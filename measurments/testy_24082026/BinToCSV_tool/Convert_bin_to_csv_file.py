#!/usr/bin/env python
#-*- coding:utf-8 –*-
#-----------------------------------------------------------------------------
# $Header: $
#-----------------------------------------------------------------------------
# Python Version:       3.4.3		 
#
# Authors:		Angel.Xu
#
# Started:		2018.07.12
#
# Copyright 2013-2018 Siglent Corporation. All Rights Reserved.
#
#-----------------------------------------------------------------------------

import json
import struct
from pathlib import Path

import numpy as np

INT_BYTE_LEN = 4
DOUBLE_BYTE_LEN = 8
RESERVE_BYTE_LEN = 0x800-0x11c

HORI_DIV_NUM = 10#14
VERT_DIV_CODE = 30#25
Magnitude = [10e-24,10e-21,10e-18,10e-15,\
             10e-12,10e-9,10e-6,10e-3,1,\
             10e3,10e6,10e9,10e12,10e15]

def deal_to_data_unit(f, para_num):
    if para_num == 1:
        stream = f.read(DOUBLE_BYTE_LEN)
        data = struct.unpack('d',stream)[0]
        stream = f.read(INT_BYTE_LEN)
        unit = struct.unpack('i',stream)[0]
        stream = f.read(INT_BYTE_LEN)
        para = data*Magnitude[unit]
    else:
        para = []
        for i in range(0,para_num):
            stream = f.read(DOUBLE_BYTE_LEN)
            data = struct.unpack('d',stream)[0]
            stream = f.read(INT_BYTE_LEN)
            unit = struct.unpack('i',stream)[0]
            stream = f.read(INT_BYTE_LEN)
            data_unit = data*Magnitude[unit]
            para.append(data_unit)
    return para
    
def deal_to_int(f, para_num):
    if para_num == 1:
        stream = f.read(INT_BYTE_LEN)
        para = struct.unpack('i',stream)[0]
    else:
        para = []
        for i in range(0,para_num):
            stream = f.read(INT_BYTE_LEN)
            data= struct.unpack('i',stream)[0]
            para.append(data)
    return para

def main(file, out_path=None):
    if out_path is None:
        out_path = Path(file).with_suffix('.npz')
    try:
        f = open(file, 'rb+')
        # SDS2000X Plus header variant: 4-byte leading field before ch_state, and a
        # vendor-specific block (channel probe/coupling info, undocumented in this
        # tool) between ch_ofst and the horizontal/wave_len block. The fields below
        # were located by pattern-matching known values (wave_len against file size,
        # sara against a plausible sample rate) directly in the header bytes rather
        # than by walking the SDS5000X-shaped layout this script was written for.
        f.read(INT_BYTE_LEN)  # leading field, unused
        ch_state = deal_to_int(f, 4)
        ch_vdiv = deal_to_data_unit(f, 4)
        ch_ofst = deal_to_data_unit(f, 4)
        # deal_to_data_unit over-reads 4 bytes per item on this header variant, so every
        # field past the first channel is misaligned. Only ch_vdiv[0]/ch_ofst[0] (the only
        # active channel here) land correctly; everything below is re-anchored by absolute
        # seek instead of relying on the cumulative (buggy) read position.
        f.seek(0x1D0)
        hori_0 = deal_to_data_unit(f, 1)
        hori_list = [hori_0, 0.0]
        f.seek(0x1E8)
        wave_len = deal_to_int(f, 1)
        print(wave_len)
        f.seek(0x1EC)
        sara = deal_to_data_unit(f, 1)
        f.seek(0x1F8)
        di_wave_len = deal_to_int(f, 1)
        print(di_wave_len)
        f.seek(0x1FC)
        di_sara = deal_to_data_unit(f, 1)
        f.seek(0x800)
        data = f.read()
    except IOError:
        print("Error: Can't find the bin file or read failed!")
    else:
        f.close()
        print('Read data from bin file finished!')
    

    ##-------------------------convert active channels, write to npz------------------------------
    print('analog converting...')
    raw = np.frombuffer(data, dtype=np.uint8)
    channels = []
    ch_state_num = 0
    for j in range(0, len(ch_state)):
        if ch_state[j]:
            seg = raw[ch_state_num * wave_len : (ch_state_num + 1) * wave_len].astype(np.float32)
            channels.append((seg - 128) * (ch_vdiv[j] / VERT_DIV_CODE) - ch_ofst[j])
            ch_state_num += 1

    print('Writing to npz...')
    save_kwargs = {"dt_s": np.float64(1.0 / sara), "ch_a": channels[0]}
    if len(channels) >= 2:
        save_kwargs["ch_b"] = channels[1]
    np.savez_compressed(out_path, **save_kwargs)
    Path(out_path).with_suffix(".json").write_text(json.dumps({
        "source_bin": str(file),
        "record_length": wave_len,
        "sample_rate": sara,
        "n_channels": len(channels),
    }, indent=2))


if __name__ == "__main__":
    import sys
    in_path = sys.argv[1] if len(sys.argv) > 1 else 'F:\\bin_50M\\SDS5104X_3.bin'
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(in_path, out_path)

