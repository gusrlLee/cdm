#pragma once

#include "base.h"
#include "bc1.h"
#include <cstdint>
#include <cmath>
#if !defined(__CUDACC__)
#include <algorithm>
#include <immintrin.h>
#endif

/* BC6H decode tables and bit extraction are adapted from bcdec.h v0.985.
   Copyright (c) 2022 Sergii Kudlai
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to inclusion of this notice. THE SOFTWARE IS
   PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED. */
namespace bc6h
{
using MeanImage = MeanImage3;

struct Block
{
    uint64_t low;
    uint64_t high;
};
static_assert(sizeof(Block) == 16);

CDM_INLINE int weight4(uint32_t index)
{
    switch (index)
    {
    case 0:
        return 0;
    case 1:
        return 4;
    case 2:
        return 9;
    case 3:
        return 13;
    case 4:
        return 17;
    case 5:
        return 21;
    case 6:
        return 26;
    case 7:
        return 30;
    case 8:
        return 34;
    case 9:
        return 38;
    case 10:
        return 43;
    case 11:
        return 47;
    case 12:
        return 51;
    case 13:
        return 55;
    case 14:
        return 60;
    default:
        return 64;
    }
}

CDM_INLINE uint16_t get_u16le(const void *s)
{
    unsigned short l = (unsigned short)((const unsigned char *)s)[0];
    unsigned short h = (unsigned short)((const unsigned char *)s)[1];
    return l | (h << 8);
}
CDM_INLINE uint32_t get_u32le(const void *s)
{
    unsigned int l = get_u16le(s);
    unsigned int h = get_u16le((const unsigned char *)s + 2);
    return l | (h << 16);
}
CDM_INLINE uint64_t get_u64le(const void *s)
{
    unsigned long long l = get_u32le(s);
    unsigned long long h = get_u32le((const unsigned char *)s + 4);
    return l | (h << 32);
}

struct BitStream
{
    unsigned long long low;
    unsigned long long high;
};

CDM_INLINE int read_bits(BitStream *bstream, int numBits)
{
    unsigned int mask = (1 << numBits) - 1;
    /* Read the low N bits */
    unsigned int bits = (bstream->low & mask);

    bstream->low >>= numBits;
    /* Put the low N bits of "high" into the high 64-N bits of "low". */
    bstream->low |= (bstream->high & mask) << (sizeof(bstream->high) * 8 - numBits);
    bstream->high >>= numBits;

    return bits;
}

CDM_INLINE int read_bit(BitStream *bstream)
{
    return read_bits(bstream, 1);
}

/*  reversed bits pulling, used in BC6H decoding
    why ?? just why ??? */
CDM_INLINE int read_bits_reversed(BitStream *bstream, int numBits)
{
    int bits = read_bits(bstream, numBits);
    /* Reverse the bits. */
    int result = 0;
    while (numBits--)
    {
        result <<= 1;
        result |= (bits & 1);
        bits >>= 1;
    }
    return result;
}
/* http://graphics.stanford.edu/~seander/bithacks.html#VariableSignExtend */
CDM_INLINE int extend_sign(int val, int bits)
{
    uint32_t sign = 1u << (bits - 1);
    return int((uint32_t(val) ^ sign) - sign);
}

CDM_INLINE int transform_inverse(int val, int a0, int bits, int isSigned)
{
    /* If the precision of A0 is "p" bits, then the transform algorithm is:
       B0 = (B0 + A0) & ((1 << p) - 1) */
    val = (val + a0) & ((1 << bits) - 1);
    if (isSigned)
    {
        val = extend_sign(val, bits);
    }
    return val;
}

/* pretty much copy-paste from documentation */
CDM_INLINE int unquantize(int val, int bits, int isSigned)
{
    int unq, s = 0;

    if (!isSigned)
    {
        if (bits >= 15)
        {
            unq = val;
        }
        else if (!val)
        {
            unq = 0;
        }
        else if (val == ((1 << bits) - 1))
        {
            unq = 0xFFFF;
        }
        else
        {
            unq = ((val << 16) + 0x8000) >> bits;
        }
    }
    else
    {
        if (bits >= 16)
        {
            unq = val;
        }
        else
        {
            if (val < 0)
            {
                s = 1;
                val = -val;
            }

            if (val == 0)
            {
                unq = 0;
            }
            else if (val >= ((1 << (bits - 1)) - 1))
            {
                unq = 0x7FFF;
            }
            else
            {
                unq = ((val << 15) + 0x4000) >> (bits - 1);
            }

            if (s)
            {
                unq = -unq;
            }
        }
    }
    return unq;
}

CDM_INLINE int interpolate(int a, int b, const int *weights, int index)
{
    return (a * (64 - weights[index]) + b * weights[index] + 32) >> 6;
}

CDM_INLINE uint16_t finish_unquantize(int val, int isSigned)
{
    int s;

    if (!isSigned)
    {
        return (unsigned short)((val * 31) >> 6); /* scale the magnitude by 31 / 64 */
    }
    else
    {
        val = (val < 0) ? -(((-val) * 31) >> 5) : (val * 31) >> 5; /* scale the magnitude by 31 / 32 */
        s = 0;
        if (val < 0)
        {
            s = 0x8000;
            val = -val;
        }
        return (unsigned short)(s | val);
    }
}

/* modified half_to_float_fast4 from https://gist.github.com/rygorous/2144712 */
CDM_INLINE float half_to_float(uint16_t half)
{
    typedef union {
        unsigned int u;
        float f;
    } FP32;

    const FP32 magic = {113 << 23};
    const unsigned int shifted_exp = 0x7c00 << 13;
    FP32 o;
    unsigned int exp;

    o.u = (half & 0x7fff) << 13; /* exponent/mantissa bits */
    exp = shifted_exp & o.u;     /* just the exponent */
    o.u += (127 - 15) << 23;     /* exponent adjust */

    /* handle exponent special cases */
    if (exp == shifted_exp)
    {                            /* Inf/NaN? */
        o.u += (128 - 16) << 23; /* extra exp adjust */
    }
    else if (exp == 0)
    {                   /* Zero/Denormal? */
        o.u += 1 << 23; /* extra exp adjust */
        o.f -= magic.f; /* renormalize */
    }

    o.u |= (half & 0x8000) << 16; /* sign bit */
    return o.f;
}

CDM_INLINE void decode_half(const void *compressedBlock, void *decompressedBlock, int destinationPitch, int isSigned)
{
    static char actual_bits_count[4][14] = {
        {10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16}, /*  W */
        {5, 6, 5, 4, 4, 5, 6, 5, 5, 6, 10, 9, 8, 4},        /* dR */
        {5, 6, 4, 5, 4, 5, 5, 6, 5, 6, 10, 9, 8, 4},        /* dG */
        {5, 6, 4, 4, 5, 5, 5, 5, 6, 6, 10, 9, 8, 4}         /* dB */
    };

    /* There are 32 possible partition sets for a two-region tile.
       Each 4x4 block represents a single shape.
       Here also every fix-up index has MSB bit set. */
    static unsigned char partition_sets[32][4][4] = {
        {{128, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 129}}, /*  0 */
        {{128, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 129}}, /*  1 */
        {{128, 1, 1, 1}, {0, 1, 1, 1}, {0, 1, 1, 1}, {0, 1, 1, 129}}, /*  2 */
        {{128, 0, 0, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 1, 1, 129}}, /*  3 */
        {{128, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 1, 129}}, /*  4 */
        {{128, 0, 1, 1}, {0, 1, 1, 1}, {0, 1, 1, 1}, {1, 1, 1, 129}}, /*  5 */
        {{128, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 1, 1}, {1, 1, 1, 129}}, /*  6 */
        {{128, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 1}, {0, 1, 1, 129}}, /*  7 */
        {{128, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 129}}, /*  8 */
        {{128, 0, 1, 1}, {0, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 129}}, /*  9 */
        {{128, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 1, 1}, {1, 1, 1, 129}}, /* 10 */
        {{128, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 1, 129}}, /* 11 */
        {{128, 0, 0, 1}, {0, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 129}}, /* 12 */
        {{128, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 129}}, /* 13 */
        {{128, 0, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 129}}, /* 14 */
        {{128, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 1, 129}}, /* 15 */
        {{128, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 1, 0}, {1, 1, 1, 129}}, /* 16 */
        {{128, 1, 129, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}}, /* 17 */
        {{128, 0, 0, 0}, {0, 0, 0, 0}, {129, 0, 0, 0}, {1, 1, 1, 0}}, /* 18 */
        {{128, 1, 129, 1}, {0, 0, 1, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}}, /* 19 */
        {{128, 0, 129, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}}, /* 20 */
        {{128, 0, 0, 0}, {1, 0, 0, 0}, {129, 1, 0, 0}, {1, 1, 1, 0}}, /* 21 */
        {{128, 0, 0, 0}, {0, 0, 0, 0}, {129, 0, 0, 0}, {1, 1, 0, 0}}, /* 22 */
        {{128, 1, 1, 1}, {0, 0, 1, 1}, {0, 0, 1, 1}, {0, 0, 0, 129}}, /* 23 */
        {{128, 0, 129, 1}, {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 0}}, /* 24 */
        {{128, 0, 0, 0}, {1, 0, 0, 0}, {129, 0, 0, 0}, {1, 1, 0, 0}}, /* 25 */
        {{128, 1, 129, 0}, {0, 1, 1, 0}, {0, 1, 1, 0}, {0, 1, 1, 0}}, /* 26 */
        {{128, 0, 129, 1}, {0, 1, 1, 0}, {0, 1, 1, 0}, {1, 1, 0, 0}}, /* 27 */
        {{128, 0, 0, 1}, {0, 1, 1, 1}, {129, 1, 1, 0}, {1, 0, 0, 0}}, /* 28 */
        {{128, 0, 0, 0}, {1, 1, 1, 1}, {129, 1, 1, 1}, {0, 0, 0, 0}}, /* 29 */
        {{128, 1, 129, 1}, {0, 0, 0, 1}, {1, 0, 0, 0}, {1, 1, 1, 0}}, /* 30 */
        {{128, 0, 129, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 1, 0, 0}}  /* 31 */
    };

    const int aWeight3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
    const int aWeight4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

    BitStream bstream;
    int mode, partition, numPartitions, i, j, partitionSet, indexBits, index, ep_i, actualBits0Mode;
    int r[4], g[4], b[4]; /* wxyz */
    unsigned short *decompressed;
    const int *weights;

    decompressed = (unsigned short *)decompressedBlock;

    bstream.low = get_u64le(compressedBlock);
    bstream.high = get_u64le((const unsigned char *)compressedBlock + 8);

    r[0] = r[1] = r[2] = r[3] = 0;
    g[0] = g[1] = g[2] = g[3] = 0;
    b[0] = b[1] = b[2] = b[3] = 0;

    mode = read_bits(&bstream, 2);
    if (mode > 1)
    {
        mode |= (read_bits(&bstream, 3) << 2);
    }

    /* modes >= 11 (10 in my code) are using 0 one, others will read it from the bitstream */
    partition = 0;

    switch (mode)
    {
    /* mode 1 */
    case 0b00: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 75 bits (10.555, 10.555, 10.555) */
        g[2] |= read_bit(&bstream) << 4;    /* gy[4]   */
        b[2] |= read_bit(&bstream) << 4;    /* by[4]   */
        b[3] |= read_bit(&bstream) << 4;    /* bz[4]   */
        r[0] |= read_bits(&bstream, 10);    /* rw[9:0] */
        g[0] |= read_bits(&bstream, 10);    /* gw[9:0] */
        b[0] |= read_bits(&bstream, 10);    /* bw[9:0] */
        r[1] |= read_bits(&bstream, 5);     /* rx[4:0] */
        g[3] |= read_bit(&bstream) << 4;    /* gz[4]   */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 5);     /* gx[4:0] */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        g[3] |= read_bits(&bstream, 4);     /* gz[3:0] */
        b[1] |= read_bits(&bstream, 5);     /* bx[4:0] */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 5);     /* ry[4:0] */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        r[3] |= read_bits(&bstream, 5);     /* rz[4:0] */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 0;
    }
    break;

    /* mode 2 */
    case 0b01: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 75 bits (7666, 7666, 7666) */
        g[2] |= read_bit(&bstream) << 5;    /* gy[5]   */
        g[3] |= read_bit(&bstream) << 4;    /* gz[4]   */
        g[3] |= read_bit(&bstream) << 5;    /* gz[5]   */
        r[0] |= read_bits(&bstream, 7);     /* rw[6:0] */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bit(&bstream) << 4;    /* by[4]   */
        g[0] |= read_bits(&bstream, 7);     /* gw[6:0] */
        b[2] |= read_bit(&bstream) << 5;    /* by[5]   */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        g[2] |= read_bit(&bstream) << 4;    /* gy[4]   */
        b[0] |= read_bits(&bstream, 7);     /* bw[6:0] */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        b[3] |= read_bit(&bstream) << 5;    /* bz[5]   */
        b[3] |= read_bit(&bstream) << 4;    /* bz[4]   */
        r[1] |= read_bits(&bstream, 6);     /* rx[5:0] */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 6);     /* gx[5:0] */
        g[3] |= read_bits(&bstream, 4);     /* gz[3:0] */
        b[1] |= read_bits(&bstream, 6);     /* bx[5:0] */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 6);     /* ry[5:0] */
        r[3] |= read_bits(&bstream, 6);     /* rz[5:0] */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 1;
    }
    break;

    /* mode 3 */
    case 0b00010: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 72 bits (11.555, 11.444, 11.444) */
        r[0] |= read_bits(&bstream, 10);    /* rw[9:0] */
        g[0] |= read_bits(&bstream, 10);    /* gw[9:0] */
        b[0] |= read_bits(&bstream, 10);    /* bw[9:0] */
        r[1] |= read_bits(&bstream, 5);     /* rx[4:0] */
        r[0] |= read_bit(&bstream) << 10;   /* rw[10]  */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 4);     /* gx[3:0] */
        g[0] |= read_bit(&bstream) << 10;   /* gw[10]  */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        g[3] |= read_bits(&bstream, 4);     /* gz[3:0] */
        b[1] |= read_bits(&bstream, 4);     /* bx[3:0] */
        b[0] |= read_bit(&bstream) << 10;   /* bw[10]  */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 5);     /* ry[4:0] */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        r[3] |= read_bits(&bstream, 5);     /* rz[4:0] */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 2;
    }
    break;

    /* mode 4 */
    case 0b00110: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 72 bits (11.444, 11.555, 11.444) */
        r[0] |= read_bits(&bstream, 10);    /* rw[9:0] */
        g[0] |= read_bits(&bstream, 10);    /* gw[9:0] */
        b[0] |= read_bits(&bstream, 10);    /* bw[9:0] */
        r[1] |= read_bits(&bstream, 4);     /* rx[3:0] */
        r[0] |= read_bit(&bstream) << 10;   /* rw[10]  */
        g[3] |= read_bit(&bstream) << 4;    /* gz[4]   */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 5);     /* gx[4:0] */
        g[0] |= read_bit(&bstream) << 10;   /* gw[10]  */
        g[3] |= read_bits(&bstream, 4);     /* gz[3:0] */
        b[1] |= read_bits(&bstream, 4);     /* bx[3:0] */
        b[0] |= read_bit(&bstream) << 10;   /* bw[10]  */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 4);     /* ry[3:0] */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        r[3] |= read_bits(&bstream, 4);     /* rz[3:0] */
        g[2] |= read_bit(&bstream) << 4;    /* gy[4]   */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 3;
    }
    break;

    /* mode 5 */
    case 0b01010: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 72 bits (11.444, 11.444, 11.555) */
        r[0] |= read_bits(&bstream, 10);    /* rw[9:0] */
        g[0] |= read_bits(&bstream, 10);    /* gw[9:0] */
        b[0] |= read_bits(&bstream, 10);    /* bw[9:0] */
        r[1] |= read_bits(&bstream, 4);     /* rx[3:0] */
        r[0] |= read_bit(&bstream) << 10;   /* rw[10]  */
        b[2] |= read_bit(&bstream) << 4;    /* by[4]   */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 4);     /* gx[3:0] */
        g[0] |= read_bit(&bstream) << 10;   /* gw[10]  */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        g[3] |= read_bits(&bstream, 4);     /* gz[3:0] */
        b[1] |= read_bits(&bstream, 5);     /* bx[4:0] */
        b[0] |= read_bit(&bstream) << 10;   /* bw[10]  */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 4);     /* ry[3:0] */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        r[3] |= read_bits(&bstream, 4);     /* rz[3:0] */
        b[3] |= read_bit(&bstream) << 4;    /* bz[4]   */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 4;
    }
    break;

    /* mode 6 */
    case 0b01110: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 72 bits (9555, 9555, 9555) */
        r[0] |= read_bits(&bstream, 9);     /* rw[8:0] */
        b[2] |= read_bit(&bstream) << 4;    /* by[4]   */
        g[0] |= read_bits(&bstream, 9);     /* gw[8:0] */
        g[2] |= read_bit(&bstream) << 4;    /* gy[4]   */
        b[0] |= read_bits(&bstream, 9);     /* bw[8:0] */
        b[3] |= read_bit(&bstream) << 4;    /* bz[4]   */
        r[1] |= read_bits(&bstream, 5);     /* rx[4:0] */
        g[3] |= read_bit(&bstream) << 4;    /* gz[4]   */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 5);     /* gx[4:0] */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        g[3] |= read_bits(&bstream, 4);     /* gx[3:0] */
        b[1] |= read_bits(&bstream, 5);     /* bx[4:0] */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 5);     /* ry[4:0] */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        r[3] |= read_bits(&bstream, 5);     /* rz[4:0] */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 5;
    }
    break;

    /* mode 7 */
    case 0b10010: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 72 bits (8666, 8555, 8555) */
        r[0] |= read_bits(&bstream, 8);     /* rw[7:0] */
        g[3] |= read_bit(&bstream) << 4;    /* gz[4]   */
        b[2] |= read_bit(&bstream) << 4;    /* by[4]   */
        g[0] |= read_bits(&bstream, 8);     /* gw[7:0] */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        g[2] |= read_bit(&bstream) << 4;    /* gy[4]   */
        b[0] |= read_bits(&bstream, 8);     /* bw[7:0] */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        b[3] |= read_bit(&bstream) << 4;    /* bz[4]   */
        r[1] |= read_bits(&bstream, 6);     /* rx[5:0] */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 5);     /* gx[4:0] */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        g[3] |= read_bits(&bstream, 4);     /* gz[3:0] */
        b[1] |= read_bits(&bstream, 5);     /* bx[4:0] */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 6);     /* ry[5:0] */
        r[3] |= read_bits(&bstream, 6);     /* rz[5:0] */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 6;
    }
    break;

    /* mode 8 */
    case 0b10110: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 72 bits (8555, 8666, 8555) */
        r[0] |= read_bits(&bstream, 8);     /* rw[7:0] */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        b[2] |= read_bit(&bstream) << 4;    /* by[4]   */
        g[0] |= read_bits(&bstream, 8);     /* gw[7:0] */
        g[2] |= read_bit(&bstream) << 5;    /* gy[5]   */
        g[2] |= read_bit(&bstream) << 4;    /* gy[4]   */
        b[0] |= read_bits(&bstream, 8);     /* bw[7:0] */
        g[3] |= read_bit(&bstream) << 5;    /* gz[5]   */
        b[3] |= read_bit(&bstream) << 4;    /* bz[4]   */
        r[1] |= read_bits(&bstream, 5);     /* rx[4:0] */
        g[3] |= read_bit(&bstream) << 4;    /* gz[4]   */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 6);     /* gx[5:0] */
        g[3] |= read_bits(&bstream, 4);     /* zx[3:0] */
        b[1] |= read_bits(&bstream, 5);     /* bx[4:0] */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 5);     /* ry[4:0] */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        r[3] |= read_bits(&bstream, 5);     /* rz[4:0] */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 7;
    }
    break;

    /* mode 9 */
    case 0b11010: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 72 bits (8555, 8555, 8666) */
        r[0] |= read_bits(&bstream, 8);     /* rw[7:0] */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bit(&bstream) << 4;    /* by[4]   */
        g[0] |= read_bits(&bstream, 8);     /* gw[7:0] */
        b[2] |= read_bit(&bstream) << 5;    /* by[5]   */
        g[2] |= read_bit(&bstream) << 4;    /* gy[4]   */
        b[0] |= read_bits(&bstream, 8);     /* bw[7:0] */
        b[3] |= read_bit(&bstream) << 5;    /* bz[5]   */
        b[3] |= read_bit(&bstream) << 4;    /* bz[4]   */
        r[1] |= read_bits(&bstream, 5);     /* bw[4:0] */
        g[3] |= read_bit(&bstream) << 4;    /* gz[4]   */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 5);     /* gx[4:0] */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        g[3] |= read_bits(&bstream, 4);     /* gz[3:0] */
        b[1] |= read_bits(&bstream, 6);     /* bx[5:0] */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 5);     /* ry[4:0] */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        r[3] |= read_bits(&bstream, 5);     /* rz[4:0] */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 8;
    }
    break;

    /* mode 10 */
    case 0b11110: {
        /* Partitition indices: 46 bits
           Partition: 5 bits
           Color Endpoints: 72 bits (6666, 6666, 6666) */
        r[0] |= read_bits(&bstream, 6);     /* rw[5:0] */
        g[3] |= read_bit(&bstream) << 4;    /* gz[4]   */
        b[3] |= read_bit(&bstream);         /* bz[0]   */
        b[3] |= read_bit(&bstream) << 1;    /* bz[1]   */
        b[2] |= read_bit(&bstream) << 4;    /* by[4]   */
        g[0] |= read_bits(&bstream, 6);     /* gw[5:0] */
        g[2] |= read_bit(&bstream) << 5;    /* gy[5]   */
        b[2] |= read_bit(&bstream) << 5;    /* by[5]   */
        b[3] |= read_bit(&bstream) << 2;    /* bz[2]   */
        g[2] |= read_bit(&bstream) << 4;    /* gy[4]   */
        b[0] |= read_bits(&bstream, 6);     /* bw[5:0] */
        g[3] |= read_bit(&bstream) << 5;    /* gz[5]   */
        b[3] |= read_bit(&bstream) << 3;    /* bz[3]   */
        b[3] |= read_bit(&bstream) << 5;    /* bz[5]   */
        b[3] |= read_bit(&bstream) << 4;    /* bz[4]   */
        r[1] |= read_bits(&bstream, 6);     /* rx[5:0] */
        g[2] |= read_bits(&bstream, 4);     /* gy[3:0] */
        g[1] |= read_bits(&bstream, 6);     /* gx[5:0] */
        g[3] |= read_bits(&bstream, 4);     /* gz[3:0] */
        b[1] |= read_bits(&bstream, 6);     /* bx[5:0] */
        b[2] |= read_bits(&bstream, 4);     /* by[3:0] */
        r[2] |= read_bits(&bstream, 6);     /* ry[5:0] */
        r[3] |= read_bits(&bstream, 6);     /* rz[5:0] */
        partition = read_bits(&bstream, 5); /* d[4:0]  */
        mode = 9;
    }
    break;

    /* mode 11 */
    case 0b00011: {
        /* Partitition indices: 63 bits
           Partition: 0 bits
           Color Endpoints: 60 bits (10.10, 10.10, 10.10) */
        r[0] |= read_bits(&bstream, 10); /* rw[9:0] */
        g[0] |= read_bits(&bstream, 10); /* gw[9:0] */
        b[0] |= read_bits(&bstream, 10); /* bw[9:0] */
        r[1] |= read_bits(&bstream, 10); /* rx[9:0] */
        g[1] |= read_bits(&bstream, 10); /* gx[9:0] */
        b[1] |= read_bits(&bstream, 10); /* bx[9:0] */
        mode = 10;
    }
    break;

    /* mode 12 */
    case 0b00111: {
        /* Partitition indices: 63 bits
           Partition: 0 bits
           Color Endpoints: 60 bits (11.9, 11.9, 11.9) */
        r[0] |= read_bits(&bstream, 10);  /* rw[9:0] */
        g[0] |= read_bits(&bstream, 10);  /* gw[9:0] */
        b[0] |= read_bits(&bstream, 10);  /* bw[9:0] */
        r[1] |= read_bits(&bstream, 9);   /* rx[8:0] */
        r[0] |= read_bit(&bstream) << 10; /* rw[10]  */
        g[1] |= read_bits(&bstream, 9);   /* gx[8:0] */
        g[0] |= read_bit(&bstream) << 10; /* gw[10]  */
        b[1] |= read_bits(&bstream, 9);   /* bx[8:0] */
        b[0] |= read_bit(&bstream) << 10; /* bw[10]  */
        mode = 11;
    }
    break;

    /* mode 13 */
    case 0b01011: {
        /* Partitition indices: 63 bits
           Partition: 0 bits
           Color Endpoints: 60 bits (12.8, 12.8, 12.8) */
        r[0] |= read_bits(&bstream, 10);               /* rw[9:0] */
        g[0] |= read_bits(&bstream, 10);               /* gw[9:0] */
        b[0] |= read_bits(&bstream, 10);               /* bw[9:0] */
        r[1] |= read_bits(&bstream, 8);                /* rx[7:0] */
        r[0] |= read_bits_reversed(&bstream, 2) << 10; /* rx[10:11] */
        g[1] |= read_bits(&bstream, 8);                /* gx[7:0] */
        g[0] |= read_bits_reversed(&bstream, 2) << 10; /* gx[10:11] */
        b[1] |= read_bits(&bstream, 8);                /* bx[7:0] */
        b[0] |= read_bits_reversed(&bstream, 2) << 10; /* bx[10:11] */
        mode = 12;
    }
    break;

    /* mode 14 */
    case 0b01111: {
        /* Partitition indices: 63 bits
           Partition: 0 bits
           Color Endpoints: 60 bits (16.4, 16.4, 16.4) */
        r[0] |= read_bits(&bstream, 10);               /* rw[9:0] */
        g[0] |= read_bits(&bstream, 10);               /* gw[9:0] */
        b[0] |= read_bits(&bstream, 10);               /* bw[9:0] */
        r[1] |= read_bits(&bstream, 4);                /* rx[3:0] */
        r[0] |= read_bits_reversed(&bstream, 6) << 10; /* rw[10:15] */
        g[1] |= read_bits(&bstream, 4);                /* gx[3:0] */
        g[0] |= read_bits_reversed(&bstream, 6) << 10; /* gw[10:15] */
        b[1] |= read_bits(&bstream, 4);                /* bx[3:0] */
        b[0] |= read_bits_reversed(&bstream, 6) << 10; /* bw[10:15] */
        mode = 13;
    }
    break;

    default: {
        /* Modes 10011, 10111, 11011, and 11111 (not shown) are reserved.
           Do not use these in your encoder. If the hardware is passed blocks
           with one of these modes specified, the resulting decompressed block
           must contain all zeroes in all channels except for the alpha channel. */
        for (i = 0; i < 4; ++i)
        {
            for (j = 0; j < 4; ++j)
            {
                decompressed[j * 3 + 0] = 0;
                decompressed[j * 3 + 1] = 0;
                decompressed[j * 3 + 2] = 0;
            }
            decompressed += destinationPitch;
        }

        return;
    }
    }

    numPartitions = (mode >= 10) ? 0 : 1;

    actualBits0Mode = actual_bits_count[0][mode];
    if (isSigned)
    {
        r[0] = extend_sign(r[0], actualBits0Mode);
        g[0] = extend_sign(g[0], actualBits0Mode);
        b[0] = extend_sign(b[0], actualBits0Mode);
    }

    /* Mode 11 (like Mode 10) does not use delta compression,
       and instead stores both color endpoints explicitly.  */
    if ((mode != 9 && mode != 10) || isSigned)
    {
        for (i = 1; i < (numPartitions + 1) * 2; ++i)
        {
            r[i] = extend_sign(r[i], actual_bits_count[1][mode]);
            g[i] = extend_sign(g[i], actual_bits_count[2][mode]);
            b[i] = extend_sign(b[i], actual_bits_count[3][mode]);
        }
    }

    if (mode != 9 && mode != 10)
    {
        for (i = 1; i < (numPartitions + 1) * 2; ++i)
        {
            r[i] = transform_inverse(r[i], r[0], actualBits0Mode, isSigned);
            g[i] = transform_inverse(g[i], g[0], actualBits0Mode, isSigned);
            b[i] = transform_inverse(b[i], b[0], actualBits0Mode, isSigned);
        }
    }

    for (i = 0; i < (numPartitions + 1) * 2; ++i)
    {
        r[i] = unquantize(r[i], actualBits0Mode, isSigned);
        g[i] = unquantize(g[i], actualBits0Mode, isSigned);
        b[i] = unquantize(b[i], actualBits0Mode, isSigned);
    }

    weights = (mode >= 10) ? aWeight4 : aWeight3;
    for (i = 0; i < 4; ++i)
    {
        for (j = 0; j < 4; ++j)
        {
            partitionSet = (mode >= 10) ? ((i | j) ? 0 : 128) : partition_sets[partition][i][j];

            indexBits = (mode >= 10) ? 4 : 3;
            /* fix-up index is specified with one less bit */
            /* The fix-up index for subset 0 is always index 0 */
            if (partitionSet & 0x80)
            {
                indexBits--;
            }
            partitionSet &= 0x01;

            index = read_bits(&bstream, indexBits);

            ep_i = partitionSet * 2;
            decompressed[j * 3 + 0] = finish_unquantize(interpolate(r[ep_i], r[ep_i + 1], weights, index), isSigned);
            decompressed[j * 3 + 1] = finish_unquantize(interpolate(g[ep_i], g[ep_i + 1], weights, index), isSigned);
            decompressed[j * 3 + 2] = finish_unquantize(interpolate(b[ep_i], b[ep_i + 1], weights, index), isSigned);
        }

        decompressed += destinationPitch;
    }
}

CDM_INLINE uint16_t float_to_half(float value)
{
    union {
        float f;
        uint32_t u;
    } in = {value};
    uint32_t sign = (in.u >> 16) & 0x8000u;
    uint32_t magnitude = in.u & 0x7fffffffu;
    if (magnitude >= 0x477fe000u)
        return uint16_t(sign | 0x7bffu);
    if (magnitude < 0x33000000u)
        return uint16_t(sign);
    uint32_t exponent = magnitude >> 23;
    uint32_t mantissa = magnitude & 0x7fffffu;
    if (exponent <= 112)
    {
        mantissa |= 0x800000u;
        uint32_t shift = 126u - exponent;
        uint32_t rounded = (mantissa + ((1u << (shift - 1)) - 1u) + ((mantissa >> shift) & 1u)) >> shift;
        return uint16_t(sign | rounded);
    }
    uint32_t rounded = magnitude + 0xfffu + ((magnitude >> 13) & 1u);
    return uint16_t(sign | ((rounded - 0x38000000u) >> 13));
}

CDM_INLINE void decode_block(const bc6h::Block &block, Float3 pixels[16])
{
    uint16_t half[48];
    decode_half(&block, half, 12, 0);
    for (int i = 0; i < 16; ++i)
        pixels[i] = {half_to_float(half[i * 3]), half_to_float(half[i * 3 + 1]), half_to_float(half[i * 3 + 2])};
}

CDM_INLINE float sanitize_hdr(float value);
CDM_INLINE Float3 sanitize_hdr(const Float3 &value);

CDM_INLINE void get_quadrant_means(const bc6h::Block &block, Float3 out[4])
{
    Float3 p[16];
    decode_block(block, p);
    for (int i = 0; i < 16; ++i)
        p[i] = sanitize_hdr(p[i]);
    for (uint32_t q = 0; q < 4; ++q)
    {
        uint32_t x = (q & 1u) * 2u, y = (q >> 1u) * 2u;
        out[q] = (p[y * 4 + x] + p[y * 4 + x + 1] + p[(y + 1) * 4 + x] + p[(y + 1) * 4 + x + 1]) * 0.25f;
    }
}

CDM_INLINE int unquantize10(uint32_t code)
{
    return code == 0 ? 0 : (code == 1023 ? 0xffff : int((code * 65536u + 0x8000u) >> 10));
}

CDM_INLINE uint16_t endpoint_half(uint32_t code)
{
    return uint16_t((unquantize10(code) * 31) >> 6);
}

CDM_INLINE float endpoint_float(uint32_t code)
{
    return half_to_float(endpoint_half(code));
}

CDM_INLINE float sanitize_hdr(float value)
{
    return isfinite(value) ? (value < 0.0f ? 0.0f : (value > 65504.0f ? 65504.0f : value)) : 0.0f;
}

CDM_INLINE Float3 sanitize_hdr(const Float3 &value)
{
    return {sanitize_hdr(value.r), sanitize_hdr(value.g), sanitize_hdr(value.b)};
}

CDM_INLINE uint32_t quantize_endpoint(float value)
{
    value = sanitize_hdr(value);
    uint32_t lo = 0, hi = 1023;
    while (lo < hi)
    {
        uint32_t mid = (lo + hi) >> 1;
        if (endpoint_float(mid) < value)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (lo == 0)
        return 0;
    float lower_error = value - endpoint_float(lo - 1);
    float upper_error = endpoint_float(lo) - value;
    return lower_error <= upper_error ? lo - 1 : lo;
}

CDM_INLINE bool hdr_covariance_degenerate(const SymMat3 &cov, const Float3 &mean)
{
    float scale = fmaxf(fmaxf(mean.r, mean.g), mean.b);
    scale = fmaxf(scale, 5.96046448e-8f);
    return cov.rr + cov.gg + cov.bb <= scale * scale * 1e-12f;
}

CDM_INLINE Float3 palette_color(const uint32_t endpoint[2][3], uint32_t selector)
{
    float value[3];
    for (int channel = 0; channel < 3; ++channel)
    {
        int a = unquantize10(endpoint[0][channel]);
        int b = unquantize10(endpoint[1][channel]);
        int weight = weight4(selector);
        uint16_t h = finish_unquantize((a * (64 - weight) + b * weight + 32) >> 6, 0);
        value[channel] = half_to_float(h);
    }
    return {value[0], value[1], value[2]};
}

CDM_INLINE bc6h::Block pack_mode11(uint32_t endpoint[2][3], uint32_t selector[16])
{
    if (selector[0] >= 8)
    {
        for (int c = 0; c < 3; ++c)
        {
            uint32_t t = endpoint[0][c];
            endpoint[0][c] = endpoint[1][c];
            endpoint[1][c] = t;
        }
        for (int i = 0; i < 16; ++i)
            selector[i] = 15u - selector[i];
    }
    uint64_t low = 3u;
    low |= uint64_t(endpoint[0][0]) << 5;
    low |= uint64_t(endpoint[0][1]) << 15;
    low |= uint64_t(endpoint[0][2]) << 25;
    low |= uint64_t(endpoint[1][0]) << 35;
    low |= uint64_t(endpoint[1][1]) << 45;
    low |= uint64_t(endpoint[1][2] & 0x1ffu) << 55;
    uint64_t high = uint64_t(endpoint[1][2] >> 9) | (uint64_t(selector[0]) << 1);
    for (int i = 1; i < 16; ++i)
        high |= uint64_t(selector[i]) << (4 * i);
    return {low, high};
}

CDM_INLINE bc6h::Block encode_samples_scalar(const Float3 samples[16])
{
    Float3 clean[16];
    Float3 mean = Float3::zero();
    for (int i = 0; i < 16; ++i)
    {
        clean[i] = sanitize_hdr(samples[i]);
        mean += clean[i];
    }
    mean = mean * (1.0f / 16.0f);
    SymMat3 cov = SymMat3::zero();
    for (int i = 0; i < 16; ++i)
        cov.accumulate_outer(clean[i] - mean, 1.0f / 16.0f);
    Float3 axis = compute_principal_axis(cov);
    float minimum = 1e30f, maximum = -1e30f;
    for (int i = 0; i < 16; ++i)
    {
        float p = dot(clean[i] - mean, axis);
        minimum = p < minimum ? p : minimum;
        maximum = p > maximum ? p : maximum;
    }
    Float3 p[2] = {mean + axis * minimum, mean + axis * maximum};
    if (hdr_covariance_degenerate(cov, mean))
        p[0] = p[1] = mean;
    float component[2][3] = {{p[0].r, p[0].g, p[0].b}, {p[1].r, p[1].g, p[1].b}};
    uint32_t endpoint[2][3];
    for (int e = 0; e < 2; ++e)
        for (int c = 0; c < 3; ++c)
            endpoint[e][c] = quantize_endpoint(component[e][c]);
    Float3 palette[16];
    for (uint32_t i = 0; i < 16; ++i)
        palette[i] = palette_color(endpoint, i);
    uint32_t selector[16];
    for (int i = 0; i < 16; ++i)
    {
        float best = 1e30f;
        uint32_t best_index = 0;
        for (uint32_t j = 0; j < 16; ++j)
        {
            float error = length_sq(clean[i] - palette[j]);
            if (error < best)
            {
                best = error;
                best_index = j;
            }
        }
        selector[texel_map[i]] = best_index;
    }
    return pack_mode11(endpoint, selector);
}

CDM_INLINE bc6h::Block generate_child_block_scalar(const bc6h::Block &p00, const bc6h::Block &p10,
                                                   const bc6h::Block &p01, const bc6h::Block &p11,
                                                   Float3 parent_means[4], uint32_t valid_width, uint32_t valid_height)
{
    Float3 samples[16];
    get_quadrant_means(p00, samples);
    get_quadrant_means(p10, samples + 4);
    get_quadrant_means(p01, samples + 8);
    get_quadrant_means(p11, samples + 12);
    for (int parent = 0; parent < 4; ++parent)
        parent_means[parent] =
            (samples[parent * 4] + samples[parent * 4 + 1] + samples[parent * 4 + 2] + samples[parent * 4 + 3]) * 0.25f;
    repeat_small_samples(samples, valid_width, valid_height);
    return encode_samples_scalar(samples);
}

CDM_INLINE bc6h::Block generate_child_block_from_means_scalar(const MeanImage &source, uint32_t block_x,
                                                              uint32_t block_y, uint32_t valid_width,
                                                              uint32_t valid_height)
{
    Float3 samples[16];
    for (uint32_t i = 0; i < 16; ++i)
    {
        uint32_t local = texel_map[i];
        samples[i] = source.get(repeat_small_coordinate(block_x * 4 + (local & 3u), valid_width),
                                repeat_small_coordinate(block_y * 4 + (local >> 2), valid_height));
    }
    return encode_samples_scalar(samples);
}

#if !defined(__CUDACC__)
CDM_INLINE void encode_samples_x4(const Float3x4 samples[16], bc6h::Block output[4])
{
    Float3x4 clean[16];
    const __m128 zero = _mm_setzero_ps(), maximum_hdr = _mm_set1_ps(65504.0f);
    for (int i = 0; i < 16; ++i)
    {
        auto clean4 = [&](const v4f &v) {
            return v4f(_mm_min_ps(_mm_max_ps(_mm_and_ps(v.v, _mm_cmpord_ps(v.v, v.v)), zero), maximum_hdr));
        };
        clean[i] = {clean4(samples[i].r), clean4(samples[i].g), clean4(samples[i].b)};
    }
    Float3x4 mean;
    SymMat3x4 cov;
    bc1::compute_child_moments(clean, mean, cov);
    Float3x4 axis = compute_principal_axis(cov);
    v4f minimum(1e30f), maximum(-1e30f);
    for (int i = 0; i < 16; ++i)
    {
        v4f p = dot(clean[i] - mean, axis);
        minimum = _mm_min_ps(minimum.v, p.v);
        maximum = _mm_max_ps(maximum.v, p.v);
    }
    Float3x4 p0 = mean + axis * minimum, p1 = mean + axis * maximum;
    alignas(16) float values[2][3][4];
    _mm_store_ps(values[0][0], p0.r.v);
    _mm_store_ps(values[0][1], p0.g.v);
    _mm_store_ps(values[0][2], p0.b.v);
    _mm_store_ps(values[1][0], p1.r.v);
    _mm_store_ps(values[1][1], p1.g.v);
    _mm_store_ps(values[1][2], p1.b.v);
    alignas(16) float means[3][4], covariance[6][4];
    _mm_store_ps(means[0], mean.r.v);
    _mm_store_ps(means[1], mean.g.v);
    _mm_store_ps(means[2], mean.b.v);
    _mm_store_ps(covariance[0], cov.rr.v);
    _mm_store_ps(covariance[1], cov.gg.v);
    _mm_store_ps(covariance[2], cov.bb.v);
    _mm_store_ps(covariance[3], cov.rg.v);
    _mm_store_ps(covariance[4], cov.rb.v);
    _mm_store_ps(covariance[5], cov.gb.v);
    uint32_t endpoints[4][2][3];
    Float3 lane_palette[4][16];
    for (int lane = 0; lane < 4; ++lane)
    {
        SymMat3 lane_cov = {covariance[0][lane], covariance[1][lane], covariance[2][lane],
                            covariance[3][lane], covariance[4][lane], covariance[5][lane]};
        Float3 lane_mean = {means[0][lane], means[1][lane], means[2][lane]};
        if (hdr_covariance_degenerate(lane_cov, lane_mean))
            for (int e = 0; e < 2; ++e)
                for (int c = 0; c < 3; ++c)
                    values[e][c][lane] = means[c][lane];
        for (int e = 0; e < 2; ++e)
            for (int c = 0; c < 3; ++c)
                endpoints[lane][e][c] = quantize_endpoint(values[e][c][lane]);
        for (int s = 0; s < 16; ++s)
            lane_palette[lane][s] = palette_color(endpoints[lane], s);
    }
    uint32_t selectors[4][16] = {};
    for (int i = 0; i < 16; ++i)
    {
        v4f best(1e30f);
        __m128i best_index = _mm_setzero_si128();
        for (int s = 0; s < 16; ++s)
        {
            Float3x4 color = {
                _mm_set_ps(lane_palette[3][s].r, lane_palette[2][s].r, lane_palette[1][s].r, lane_palette[0][s].r),
                _mm_set_ps(lane_palette[3][s].g, lane_palette[2][s].g, lane_palette[1][s].g, lane_palette[0][s].g),
                _mm_set_ps(lane_palette[3][s].b, lane_palette[2][s].b, lane_palette[1][s].b, lane_palette[0][s].b)};
            v4f error = length_sq(clean[i] - color);
            __m128 mask = _mm_cmplt_ps(error.v, best.v);
            best = _mm_blendv_ps(best.v, error.v, mask);
            best_index = _mm_blendv_epi8(best_index, _mm_set1_epi32(s), _mm_castps_si128(mask));
        }
        alignas(16) uint32_t index[4];
        _mm_store_si128((__m128i *)index, best_index);
        for (int lane = 0; lane < 4; ++lane)
            selectors[lane][texel_map[i]] = index[lane];
    }
    for (int lane = 0; lane < 4; ++lane)
        output[lane] = pack_mode11(endpoints[lane], selectors[lane]);
}

CDM_INLINE void generate_child_blocks_x4(const bc6h::Block *row0, const bc6h::Block *row1, bc6h::Block *output,
                                         const MeanImage &means, uint32_t source_x, uint32_t source_y0,
                                         uint32_t source_y1, uint32_t valid_width, uint32_t valid_height)
{
    Float3 scalar[4][16];
    for (int lane = 0; lane < 4; ++lane)
    {
        Float3 parent_means[4];
        get_quadrant_means(row0[lane * 2], scalar[lane]);
        get_quadrant_means(row0[lane * 2 + 1], scalar[lane] + 4);
        get_quadrant_means(row1[lane * 2], scalar[lane] + 8);
        get_quadrant_means(row1[lane * 2 + 1], scalar[lane] + 12);
        for (int p = 0; p < 4; ++p)
            parent_means[p] =
                (scalar[lane][p * 4] + scalar[lane][p * 4 + 1] + scalar[lane][p * 4 + 2] + scalar[lane][p * 4 + 3]) *
                0.25f;
        means.set(source_x + lane * 2, source_y0, parent_means[0]);
        means.set(source_x + lane * 2 + 1, source_y0, parent_means[1]);
        means.set(source_x + lane * 2, source_y1, parent_means[2]);
        means.set(source_x + lane * 2 + 1, source_y1, parent_means[3]);
        repeat_small_samples(scalar[lane], valid_width, valid_height);
    }
    Float3x4 packed[16];
    for (int i = 0; i < 16; ++i)
        packed[i] = {_mm_set_ps(scalar[3][i].r, scalar[2][i].r, scalar[1][i].r, scalar[0][i].r),
                     _mm_set_ps(scalar[3][i].g, scalar[2][i].g, scalar[1][i].g, scalar[0][i].g),
                     _mm_set_ps(scalar[3][i].b, scalar[2][i].b, scalar[1][i].b, scalar[0][i].b)};
    encode_samples_x4(packed, output);
}

CDM_INLINE void generate_child_blocks_from_means_x4(const MeanImage &source, uint32_t block_x, uint32_t block_y,
                                                    uint32_t lanes, bc6h::Block *destination, uint32_t valid_width,
                                                    uint32_t valid_height)
{
    if (lanes < 4)
    {
        for (uint32_t lane = 0; lane < lanes; ++lane)
            destination[lane] =
                generate_child_block_from_means_scalar(source, block_x + lane, block_y, valid_width, valid_height);
        return;
    }
    Float3 scalar[4][16];
    for (uint32_t lane = 0; lane < 4; ++lane)
        for (uint32_t i = 0; i < 16; ++i)
        {
            uint32_t x = block_x + (lane < lanes ? lane : lanes - 1);
            uint32_t local = texel_map[i];
            scalar[lane][i] = source.get(repeat_small_coordinate(x * 4 + (local & 3u), valid_width),
                                         repeat_small_coordinate(block_y * 4 + (local >> 2), valid_height));
        }
    Float3x4 packed[16];
    for (int i = 0; i < 16; ++i)
        packed[i] = {_mm_set_ps(scalar[3][i].r, scalar[2][i].r, scalar[1][i].r, scalar[0][i].r),
                     _mm_set_ps(scalar[3][i].g, scalar[2][i].g, scalar[1][i].g, scalar[0][i].g),
                     _mm_set_ps(scalar[3][i].b, scalar[2][i].b, scalar[1][i].b, scalar[0][i].b)};
    bc6h::Block encoded[4];
    encode_samples_x4(packed, encoded);
    for (uint32_t lane = 0; lane < lanes; ++lane)
        destination[lane] = encoded[lane];
}
#endif

#if defined(__CUDACC__)
// Decode work is shared by four lanes; encoding assigns one sample to each
// half-warp lane.
__device__ __forceinline__ Float3 device_quadrant_mean(const Block &block, uint32_t quadrant)
{
    const unsigned mask = half_warp_mask();
    const uint32_t source_lane = (threadIdx.x & 15u) & ~3u;
    Float3 means[4] = {};
    if (quadrant == 0)
    {
        Float3 pixels[16];
        decode_block(block, pixels);
        for (Float3 &pixel : pixels)
            pixel = sanitize_hdr(pixel);
        for (uint32_t q = 0; q < 4; ++q)
        {
            const uint32_t x = (q & 1u) * 2u, y = (q >> 1u) * 2u;
            means[q] =
                (pixels[y * 4 + x] + pixels[y * 4 + x + 1] + pixels[(y + 1) * 4 + x] + pixels[(y + 1) * 4 + x + 1]) *
                0.25f;
        }
    }
    Float3 result{};
    for (uint32_t q = 0; q < 4; ++q)
    {
        Float3 value = means[q];
        value = {__shfl_sync(mask, value.r, source_lane, 16), __shfl_sync(mask, value.g, source_lane, 16),
                 __shfl_sync(mask, value.b, source_lane, 16)};
        if (quadrant == q)
            result = value;
    }
    return result;
}

__device__ __forceinline__ void encode_half_warp(Float3 sample, uint32_t lane, Block *output)
{
    const unsigned mask = half_warp_mask();
    sample = sanitize_hdr(sample);
    Float3 mean;
    const SymMat3 covariance = half_warp_moments(sample, mean);
    Float3 axis = {1, 0, 0};
    if (lane == 0)
        axis = compute_principal_axis(covariance);
    axis = {__shfl_sync(mask, axis.r, 0, 16), __shfl_sync(mask, axis.g, 0, 16), __shfl_sync(mask, axis.b, 0, 16)};
    const float projection = dot(sample - mean, axis);
    Float3 p0 = mean + axis * min16(projection, mask);
    Float3 p1 = mean + axis * max16(projection, mask);
    bool degenerate = lane == 0 && hdr_covariance_degenerate(covariance, mean);
    degenerate = __shfl_sync(mask, degenerate, 0, 16);
    if (degenerate)
        p0 = p1 = mean;
    uint32_t endpoints[2][3] = {};
    if (lane == 0)
    {
        endpoints[0][0] = quantize_endpoint(p0.r);
        endpoints[0][1] = quantize_endpoint(p0.g);
        endpoints[0][2] = quantize_endpoint(p0.b);
        endpoints[1][0] = quantize_endpoint(p1.r);
        endpoints[1][1] = quantize_endpoint(p1.g);
        endpoints[1][2] = quantize_endpoint(p1.b);
    }
    for (int e = 0; e < 2; ++e)
        for (int c = 0; c < 3; ++c)
            endpoints[e][c] = __shfl_sync(mask, endpoints[e][c], 0, 16);
    const Float3 owned = palette_color(endpoints, lane);
    float best = 1e30f;
    uint32_t selector = 0;
    for (uint32_t s = 0; s < 16; ++s)
    {
        const Float3 color = {__shfl_sync(mask, owned.r, s, 16), __shfl_sync(mask, owned.g, s, 16),
                              __shfl_sync(mask, owned.b, s, 16)};
        const float error = length_sq(sample - color);
        if (error < best)
        {
            best = error;
            selector = s;
        }
    }
    const bool reverse = __shfl_sync(mask, selector >= 8, 0, 16);
    if (reverse)
    {
        selector = 15 - selector;
        for (int c = 0; c < 3; ++c)
        {
            uint32_t t = endpoints[0][c];
            endpoints[0][c] = endpoints[1][c];
            endpoints[1][c] = t;
        }
    }
    const uint32_t texel = texel_index(lane);
    uint64_t selector_bits = texel == 0 ? uint64_t(selector) << 1 : uint64_t(selector) << (4 * texel);
    selector_bits |= uint64_t(or16(uint32_t(selector_bits), mask));
    const uint32_t high_half = or16(uint32_t(selector_bits >> 32), mask);
    if (lane == 0)
    {
        const uint64_t low = 3u | (uint64_t(endpoints[0][0]) << 5) | (uint64_t(endpoints[0][1]) << 15) |
                             (uint64_t(endpoints[0][2]) << 25) | (uint64_t(endpoints[1][0]) << 35) |
                             (uint64_t(endpoints[1][1]) << 45) | (uint64_t(endpoints[1][2] & 0x1ffu) << 55);
        const uint64_t high = uint64_t(endpoints[1][2] >> 9) | uint32_t(selector_bits) | (uint64_t(high_half) << 32);
        *output = {low, high};
    }
}
#endif

} // namespace bc6h
