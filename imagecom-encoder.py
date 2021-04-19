import os
import math
import numpy as np
from scipy import fftpack
from PIL import Image
from utils import *
from huff import HuffmanTree

def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')
def quantize(block, component):
    q = load_quantization_table(component)
    return (block / q).round().astype(np.int32)
def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])
def run_length_encode(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []

    # values are binary representations of array elements using SIZE bits
    values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values

def write_to_file(filepath, dc, ac, blocks_count, tables):
    try:
        f = open(filepath, 'w')
    except FileNotFoundError as e:
        raise FileNotFoundError(
                "No such directory: {}".format(
                    os.path.dirname(filepath))) from e

    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        # 16 bits for 'table_size'
        f.write(uint_to_binstr(len(tables[table_name]), 16))

        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                # 4 bits for the 'category'
                # 4 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)
            else:
                # 4 bits for 'run_length'
                # 4 bits for 'size'
                # 8 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key[0], 4))
                f.write(uint_to_binstr(key[1], 4))
                f.write(uint_to_binstr(len(value), 8))
                f.write(value)

    # 32 bits for 'blocks_count'
    f.write(uint_to_binstr(blocks_count, 32))

    for b in range(blocks_count):
        for c in range(3):
            category = bits_required(dc[b, c])
            symbols, values = run_length_encode(ac[b, :, c])

            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']

            f.write(dc_table[category])
            f.write(int_to_binstr(dc[b, c]))

            for i in range(len(symbols)):
                f.write(ac_table[tuple(symbols[i])])
                f.write(values[i])
    f.close()

image = Image.open('Image name') #Ganti jadi lokasi fotolu
image = image.convert('YCbCr') #basically foto dibagi jadi 3 komponen, yakni Y, Cb, Cr 
npmat = np.array(image, dtype=np.uint8) #convert jadi matrix 
rows, cols = npmat.shape[0], npmat.shape[1]
rowcount,a = divmod(rows,8)
colcount,b = divmod(cols,8)

if a == b  == 0: #Kalau ukuran foto bisa dibagi 8, langsung gas
    blocks_count = rowcount*colcount
else: #kalau ukuran foto gabisa dibagi 8, diexpand pixelnya dulu supaya bisa dibagi.
    npmat = np.append(npmat,np.zeros((8-a,cols,3)),axis = 0)
    rows += 8-a
    npmat = np.append(npmat,np.zeros((rows,8-b,3)),axis = 1)
    cols += 8-b
    rowcount,a = divmod(rows,8)
    colcount,b = divmod(cols,8)
    blocks_count = rowcount*colcount

dc = np.empty((blocks_count, 3), dtype=np.int32) 
ac = np.empty((blocks_count, 63, 3), dtype=np.int32)


for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            block_index += 1

            for k in range(3):
                # split 8x8 block and center the data range on zero
                # [0, 255] --> [-128, 127]
                block = npmat[i:i+8, j:j+8, k] - 128 
                block = dct_2d(block)
                block = quantize(block,'lum' if k == 0 else 'chrom')
                block = block_to_zigzag(block)
                dc[block_index, k] = block[0]
                ac[block_index, :, k] = block[1:]

H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
H_AC_Y = HuffmanTree(flatten(run_length_encode(ac[i, :, 0])[0]for i in range(blocks_count)))
H_AC_C = HuffmanTree(flatten(run_length_encode(ac[i, :, j])[0]for i in range(blocks_count) for j in [1, 2]))
tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),'ac_y': H_AC_Y.value_to_bitstring_table(),'dc_c': H_DC_C.value_to_bitstring_table(),'ac_c': H_AC_C.value_to_bitstring_table()}
write_to_file("Your Path Name", dc, ac, blocks_count, tables)
