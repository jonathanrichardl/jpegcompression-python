import os
import math
import numpy as np
from utils import *
from scipy import fftpack
from PIL import Image
from huff import HuffmanTree
from bitarray import bitarray

def quantizelum(block):
    lum = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 3, 4, 5, 6],
                      [2, 2, 2, 2, 4, 5, 7, 9],
                      [2, 2, 2, 4, 5, 7, 9, 12],
                      [3, 3, 4, 5, 8, 10, 12, 12],
                      [4, 4, 5, 7, 10, 12, 12, 12],
                      [5, 5, 7, 9, 12, 12, 12, 12],
                      [6, 6, 9, 12, 12, 12, 12, 12]])
    return (np.divide(block,lum)).round().astype(np.int32)
def quantizechrom(block):
    chrom = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
                      [3, 4, 6, 11, 14, 12, 12, 12],
                      [5, 6, 9, 14, 12, 12, 12, 12],
                      [9, 11, 14, 12, 12, 12, 12, 12],
                      [13, 14, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12],
                      [15, 12, 12, 12, 12, 12, 12, 12]])
    return (np.divide(block,chrom)).round().astype(np.int32)

def block_to_zigzag(block):
    b = np.zeros(64,dtype = int)
    a = np.array([[ 0,1,5,6,14,15,27,28],
                  [ 2 ,4,7,13,16,26,29,42],
                  [ 3 , 8, 12 ,17 ,25, 30, 41, 43],
                  [ 9 ,11, 18 ,24, 31 ,40, 44, 53],
                  [10 ,19 ,23, 32 ,39 ,45, 52, 54],
                  [20 ,22 ,33, 38, 46, 51, 55, 60],
                  [21, 34, 37 ,47 ,50, 56 ,59 ,61],
                  [35, 36, 48 ,49, 57 ,58,62 ,63]])
    for i in range(8):
        for j in range(8):
            val = block[i,j]
            b[a[i,j]] = val
    return b

def dct_2d(image):
    return fftpack.rfft(fftpack.rfft(image.T, axis=0),axis=1)


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
            values.append(int_to_bitarray(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_bitarray(elem.item()))
            run_length = 0
    return symbols, values

def write_table(filepath,tables,filesize):
    try:
        f = open(filepath, 'w')
    except FileNotFoundError as e:
        raise FileNotFoundError(
                "No such directory: {}".format(
                    os.path.dirname(filepath))) from e
    f.write(filesize + "\n")
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        # table size
        f.write(str(len(tables[table_name]))+"\n")

        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                
                
                f.write(str(key) +"\n") # symbol
                f.write(str(value)+"\n")# codes
            else:
                f.write(str(key[0])+"," + str(key[1]) + "\n")  # symbol
                f.write(value+"\n")# codes
    f.close()

def write_to_file(filepath, dc, ac, blocks_count,tables):
    x = bitarray()
    try:
        f = open(filepath, 'wb')
    except FileNotFoundError as e:
        raise FileNotFoundError(
                "No such directory: {}".format(
                    os.path.dirname(filepath))) from e
    # 32 bits for 'blocks_count'
    f.write(uint_to_bitarray(blocks_count, 32).tobytes())

    for b in range(blocks_count):
        for c in range(3):
            category = bits_required(dc[b, c])
            symbols, values = run_length_encode(ac[b, :, c])

            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']

            x+= bitarray(dc_table[category]) + (int_to_bitarray(dc[b, c].item()))
            
            for i in range(len(symbols)):
                x += bitarray(ac_table[tuple(symbols[i])]) + values[i]
    x.tofile(f)
    f.close()


def main():
    image = Image.open('download2.jpg')
    ycbcr = image.convert('YCbCr')

    npmat = np.array(ycbcr, dtype=np.uint8)
    rows, cols = npmat.shape[0], npmat.shape[1]
    rowcount,a = divmod(rows,8)
    colcount,b = divmod(cols,8)

    if a == b  == 0:
        blocks_count = rowcount*colcount
    else:
        npmat = np.append(npmat,np.zeros((8-a,cols,3)),axis = 0)
        rows += 8-a
        npmat = np.append(npmat,np.zeros((rows,8-b,3)),axis = 1)
        cols += 8-b
        rowcount,a = divmod(rows,8)
        colcount,b = divmod(cols,8)
        blocks_count = rowcount*colcount

    # dc is the top-left cell of the block, ac are all the other cells
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
    block_index = -1
    npmat = npmat-128
    for k in range(3):
        block_index = -1
        for i in range(0, rows, 8):
            for j in range(0, cols, 8):
                try:
                    block_index += 1
                except NameError:
                    block_index = 0
                block = npmat[i:i+8, j:j+8, k] 
                dct_matrix = dct_2d(block)
                if k == 0:
                    quant_matrix = quantizelum(dct_matrix)
                else:
                    quant_matrix = quantizechrom(dct_matrix)
                zz = block_to_zigzag(quant_matrix)
                dc[block_index, k] = zz[0]
                ac[block_index, :, k] = zz[1:]

    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    H_AC_Y = HuffmanTree(
            flatten(run_length_encode(ac[i, :, 0])[0]
                    for i in range(blocks_count)))
    H_AC_C = HuffmanTree(flatten(run_length_encode(ac[i, :, j])[0]for i in range(blocks_count) for j in [1, 2]))

    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}
    write_table("table.txt", tables,str(cols) + "," + str(rows))
    write_to_file("D:\File Kuliah\Electrical Engineering\Pemrosesan\Proyek\im.bin", dc, ac, blocks_count,tables)


if __name__ == "__main__":
    main()
