#pragma once

#include "bc1.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace bc7
{
struct Block { uint64_t low, high; };
static_assert(sizeof(Block) == 16);

struct MeanImage
{
    float *r = nullptr, *g = nullptr, *b = nullptr, *a = nullptr;
    uint32_t width = 0, height = 0;
    CDM_INLINE Float4 get(uint32_t x, uint32_t y) const
    {
        x = x < width ? x : width - 1; y = y < height ? y : height - 1;
        size_t i = size_t(y) * width + x; return {r[i], g[i], b[i], a[i]};
    }
    CDM_INLINE void set(uint32_t x, uint32_t y, const Float4 &v) const
    {
        size_t i = size_t(y) * width + x; r[i]=v.r; g[i]=v.g; b[i]=v.b; a[i]=v.a;
    }
};

CDM_INLINE bool is_mode6(const Block &b) { return (b.low & 0x7fu) == 0x40u; }

CDM_INLINE uint64_t get_bits(const Block &b, uint32_t bit, uint32_t count)
{
    if (!count) return 0;
    if (bit < 64)
    {
        uint64_t value = b.low >> bit;
        if (bit + count > 64) value |= b.high << (64 - bit);
        return count == 64 ? value : value & ((uint64_t(1) << count) - 1);
    }
    return (b.high >> (bit - 64)) & ((uint64_t(1) << count) - 1);
}

CDM_INLINE void put_bits(Block &b, uint32_t bit, uint32_t count, uint64_t value)
{
    if (bit < 64)
    {
        b.low |= value << bit;
        if (bit + count > 64) b.high |= value >> (64 - bit);
    }
    else b.high |= value << (bit - 64);
}

CDM_INLINE void unpack(const Block &b, uint8_t ep[2][4], uint8_t selector[16])
{
    uint32_t bit = 7;
    for (uint32_t c=0;c<4;++c) for (uint32_t e=0;e<2;++e) { ep[e][c]=uint8_t(get_bits(b,bit,7)); bit+=7; }
    uint32_t p0=uint32_t(get_bits(b,63,1)), p1=uint32_t(get_bits(b,64,1));
    for (uint32_t c=0;c<4;++c) { ep[0][c]=uint8_t((ep[0][c]<<1)|p0); ep[1][c]=uint8_t((ep[1][c]<<1)|p1); }
    bit=65; selector[0]=uint8_t(get_bits(b,bit,3)); bit+=3;
    for (uint32_t i=1;i<16;++i) { selector[i]=uint8_t(get_bits(b,bit,4)); bit+=4; }
}

CDM_INLINE uint8_t interpolate(uint32_t a, uint32_t b, uint32_t s)
{
    uint32_t w;
    switch(s){case 0:w=0;break;case 1:w=4;break;case 2:w=9;break;case 3:w=13;break;
    case 4:w=17;break;case 5:w=21;break;case 6:w=26;break;case 7:w=30;break;
    case 8:w=34;break;case 9:w=38;break;case 10:w=43;break;case 11:w=47;break;
    case 12:w=51;break;case 13:w=55;break;case 14:w=60;break;default:w=64;break;}
    return uint8_t(((64-w)*a+w*b+32)>>6);
}

CDM_INLINE Block pack_block(const uint8_t ep8[2][4], const uint8_t selectors[16])
{
    Block out={0x40u,0}; uint32_t bit=7;
    for(uint32_t c=0;c<4;++c) for(uint32_t e=0;e<2;++e) { put_bits(out,bit,7,ep8[e][c]>>1); bit+=7; }
    put_bits(out,63,1,ep8[0][0]&1); put_bits(out,64,1,ep8[1][0]&1);
    bit=65; put_bits(out,bit,3,selectors[0]); bit+=3;
    for(uint32_t i=1;i<16;++i){put_bits(out,bit,4,selectors[i]);bit+=4;}
    return out;
}

#if !defined(__CUDACC__)
template <bool Srgb> CDM_INLINE Float4 rgba8_to_linear(const uint8_t v[4])
{
    if constexpr (Srgb) return {bc1::c_srgb_to_linear[v[0]],bc1::c_srgb_to_linear[v[1]],bc1::c_srgb_to_linear[v[2]],v[3]*(1.0f/255.0f)};
    return {v[0]*(1.0f/255.0f),v[1]*(1.0f/255.0f),v[2]*(1.0f/255.0f),v[3]*(1.0f/255.0f)};
}

template <bool Srgb> CDM_INLINE void palette(const uint8_t ep[2][4], Float4 out[16])
{
    for (uint32_t s=0;s<16;++s) { uint8_t v[4]; for(uint32_t c=0;c<4;++c) v[c]=interpolate(ep[0][c],ep[1][c],s); out[s]=rgba8_to_linear<Srgb>(v); }
}

template <bool Srgb> CDM_INLINE void quadrant_means(const Block &block, Float4 out[4])
{
    uint8_t ep[2][4], sel[16]; unpack(block,ep,sel); Float4 pal[16]; palette<Srgb>(ep,pal);
    for(uint32_t q=0;q<4;++q) { uint32_t x=(q&1)*2,y=(q>>1)*2; out[q]=(pal[sel[y*4+x]]+pal[sel[y*4+x+1]]+pal[sel[(y+1)*4+x]]+pal[sel[(y+1)*4+x+1]])*0.25f; }
}

CDM_INLINE Float4 clamp4(Float4 v)
{
    auto c=[](float x){return std::max(0.0f,std::min(1.0f,x));}; return {c(v.r),c(v.g),c(v.b),c(v.a)};
}

template <bool Srgb> CDM_INLINE uint8_t quantize_component(float target, uint32_t pbit, bool alpha)
{
    uint32_t best=pbit; float error=1e30f;
    for(uint32_t q=0;q<128;++q)
    {
        uint32_t code=(q<<1)|pbit;
        float value=(Srgb && !alpha) ? bc1::c_srgb_to_linear[code] : code*(1.0f/255.0f);
        float e=(target-value)*(target-value); if(e<error){error=e;best=code;}
    }
    return uint8_t(best);
}

template <bool Srgb> CDM_INLINE void quantize_endpoint(const Float4 &v, uint8_t out[4])
{
    float best=1e30f;
    for(uint32_t p=0;p<2;++p)
    {
        uint8_t q[4]={quantize_component<Srgb>(v.r,p,false),quantize_component<Srgb>(v.g,p,false),
                      quantize_component<Srgb>(v.b,p,false),quantize_component<Srgb>(v.a,p,true)};
        Float4 d=rgba8_to_linear<Srgb>(q)-v; float e=length_sq(d);
        if(e<best){best=e;for(int c=0;c<4;++c)out[c]=q[c];}
    }
}

template <bool Srgb> CDM_INLINE Block encode_samples(const Float4 samples[16])
{
    Float4 mean=Float4::zero(); for(int i=0;i<16;++i)mean+=samples[i]; mean=mean*(1.0f/16.0f);
    SymMat4 cov=SymMat4::zero(); for(int i=0;i<16;++i)cov.accumulate_outer(samples[i]-mean,1.0f/16.0f);
    Float4 axis=compute_principal_axis(cov); float lo=1e30f,hi=-1e30f;
    for(int i=0;i<16;++i){float p=dot(samples[i]-mean,axis);lo=std::min(lo,p);hi=std::max(hi,p);}
    Float4 endpoints[2]={clamp4(mean+axis*lo),clamp4(mean+axis*hi)}; uint8_t ep[2][4];
    quantize_endpoint<Srgb>(endpoints[0],ep[0]); quantize_endpoint<Srgb>(endpoints[1],ep[1]);
    Float4 pal[16]; palette<Srgb>(ep,pal); uint8_t selector[16]={};
    for(uint32_t i=0;i<16;++i)
    {
        uint32_t texel=bc1::texel_index(i),best=0;float error=1e30f;
        for(uint32_t s=0;s<16;++s){float e=length_sq(samples[i]-pal[s]);if(e<error){error=e;best=s;}}
        selector[texel]=uint8_t(best);
    }
    if(selector[0]>=8)
    {
        for(int c=0;c<4;++c)std::swap(ep[0][c],ep[1][c]);
        for(auto &s:selector)s=uint8_t(15-s);
    }
    return pack_block(ep,selector);
}

CDM_INLINE Block encode_samples(const Float4 s[16],bool srgb){return srgb?encode_samples<true>(s):encode_samples<false>(s);}

CDM_INLINE Block generate_child(const Block &p00,const Block &p10,const Block &p01,const Block &p11,bool srgb,
                                Float4 parent_means[4],uint32_t valid_width,uint32_t valid_height)
{
    Float4 s[16]; if(srgb){quadrant_means<true>(p00,s);quadrant_means<true>(p10,s+4);quadrant_means<true>(p01,s+8);quadrant_means<true>(p11,s+12);}
    else{quadrant_means<false>(p00,s);quadrant_means<false>(p10,s+4);quadrant_means<false>(p01,s+8);quadrant_means<false>(p11,s+12);}
    for(int p=0;p<4;++p)parent_means[p]=(s[p*4]+s[p*4+1]+s[p*4+2]+s[p*4+3])*0.25f;
    if(valid_width<4||valid_height<4){Float4 copy[16];for(int i=0;i<16;++i)copy[i]=s[i];for(uint32_t i=0;i<16;++i)s[i]=copy[bc1::repeat_small_sample(i,valid_width,valid_height)];}
    return encode_samples(s,srgb);
}

CDM_INLINE Block generate_from_means(const MeanImage &source,uint32_t bx,uint32_t by,bool srgb,uint32_t vw,uint32_t vh)
{
    Float4 s[16];for(uint32_t i=0;i<16;++i){uint32_t t=bc1::texel_index(i);s[i]=source.get(bc1::repeat_small_coordinate(bx*4+(t&3),vw),bc1::repeat_small_coordinate(by*4+(t>>2),vh));}
    return encode_samples(s,srgb);
}

using Float4x4=Vec4T<v4f>;
using SymMat4x4=SymMat4T<v4f>;

CDM_INLINE Float4x4 principal_axis_x4(const SymMat4x4 &cov)
{
    Float4x4 axis={v4f(.5f),v4f(.5f),v4f(.5f),v4f(.5f)};
    for(int i=0;i<4;++i)
    {
        Float4x4 next=cov.multiply(axis);v4f n=length_sq(next);
        __m128 flat=_mm_cmple_ps(n.v,_mm_set1_ps(1e-20f));
        v4f inv=_mm_rsqrt_ps(_mm_blendv_ps(n.v,_mm_set1_ps(1),flat));
        axis=next*inv;
        axis.r=_mm_blendv_ps(axis.r.v,_mm_set1_ps(1),flat);
        axis.g=_mm_blendv_ps(axis.g.v,_mm_setzero_ps(),flat);
        axis.b=_mm_blendv_ps(axis.b.v,_mm_setzero_ps(),flat);
        axis.a=_mm_blendv_ps(axis.a.v,_mm_setzero_ps(),flat);
    }
    return axis;
}

template<bool Srgb> CDM_INLINE Block encode_with_endpoints(const Float4 samples[16],Float4 p0,Float4 p1)
{
    uint8_t ep[2][4];quantize_endpoint<Srgb>(clamp4(p0),ep[0]);quantize_endpoint<Srgb>(clamp4(p1),ep[1]);
    Float4 pal[16];palette<Srgb>(ep,pal);uint8_t selector[16]={};
    for(uint32_t i=0;i<16;++i){uint32_t best=0;float error=1e30f;for(uint32_t s=0;s<16;++s){float e=length_sq(samples[i]-pal[s]);if(e<error){error=e;best=s;}}selector[bc1::texel_index(i)]=uint8_t(best);}
    if(selector[0]>=8){for(int c=0;c<4;++c)std::swap(ep[0][c],ep[1][c]);for(auto &s:selector)s=uint8_t(15-s);}
    return pack_block(ep,selector);
}

template<bool Srgb> CDM_INLINE void encode_samples_x4(const Float4 scalar[4][16],Block out[4])
{
    Float4x4 samples[16];
    for(int i=0;i<16;++i){samples[i].r=_mm_set_ps(scalar[3][i].r,scalar[2][i].r,scalar[1][i].r,scalar[0][i].r);
        samples[i].g=_mm_set_ps(scalar[3][i].g,scalar[2][i].g,scalar[1][i].g,scalar[0][i].g);
        samples[i].b=_mm_set_ps(scalar[3][i].b,scalar[2][i].b,scalar[1][i].b,scalar[0][i].b);
        samples[i].a=_mm_set_ps(scalar[3][i].a,scalar[2][i].a,scalar[1][i].a,scalar[0][i].a);}
    Float4x4 mean=Float4x4::zero();for(auto &s:samples)mean+=s;mean=mean*v4f(1.0f/16);
    SymMat4x4 cov=SymMat4x4::zero();for(auto &s:samples)cov.accumulate_outer(s-mean,v4f(1.0f/16));
    Float4x4 axis=principal_axis_x4(cov);v4f lo(1e30f),hi(-1e30f);
    for(auto &s:samples){v4f x=dot(s-mean,axis);lo=_mm_min_ps(lo.v,x.v);hi=_mm_max_ps(hi.v,x.v);}
    Float4x4 p0=mean+axis*lo,p1=mean+axis*hi;alignas(16) float values[8][4];
    _mm_store_ps(values[0],p0.r.v);_mm_store_ps(values[1],p0.g.v);_mm_store_ps(values[2],p0.b.v);_mm_store_ps(values[3],p0.a.v);
    _mm_store_ps(values[4],p1.r.v);_mm_store_ps(values[5],p1.g.v);_mm_store_ps(values[6],p1.b.v);_mm_store_ps(values[7],p1.a.v);
    for(int lane=0;lane<4;++lane)out[lane]=encode_with_endpoints<Srgb>(scalar[lane],
        {values[0][lane],values[1][lane],values[2][lane],values[3][lane]},
        {values[4][lane],values[5][lane],values[6][lane],values[7][lane]});
}

CDM_INLINE void generate_from_means_x4(const MeanImage &m,uint32_t bx,uint32_t by,uint32_t lanes,Block *out,bool srgb,uint32_t vw,uint32_t vh)
{
    Float4 s[4][16]={};
    for(uint32_t lane=0;lane<lanes;++lane)for(uint32_t i=0;i<16;++i){uint32_t t=bc1::texel_index(i);s[lane][i]=m.get(
        bc1::repeat_small_coordinate((bx+lane)*4+(t&3),vw),bc1::repeat_small_coordinate(by*4+(t>>2),vh));}
    for(uint32_t lane=lanes;lane<4;++lane)for(int i=0;i<16;++i)s[lane][i]=s[0][i];
    Block encoded[4];if(srgb)encode_samples_x4<true>(s,encoded);else encode_samples_x4<false>(s,encoded);
    for(uint32_t lane=0;lane<lanes;++lane)out[lane]=encoded[lane];
}

CDM_INLINE void generate_child_x4(const Block *row0,const Block *row1,Block *out,const MeanImage &means,
    uint32_t source_x,uint32_t y0,uint32_t y1,bool srgb,uint32_t vw,uint32_t vh)
{
    Float4 s[4][16];
    for(uint32_t lane=0;lane<4;++lane)
    {
        if(srgb){quadrant_means<true>(row0[lane*2],s[lane]);quadrant_means<true>(row0[lane*2+1],s[lane]+4);
            quadrant_means<true>(row1[lane*2],s[lane]+8);quadrant_means<true>(row1[lane*2+1],s[lane]+12);}
        else{quadrant_means<false>(row0[lane*2],s[lane]);quadrant_means<false>(row0[lane*2+1],s[lane]+4);
            quadrant_means<false>(row1[lane*2],s[lane]+8);quadrant_means<false>(row1[lane*2+1],s[lane]+12);}
        for(uint32_t p=0;p<4;++p){Float4 mean=(s[lane][p*4]+s[lane][p*4+1]+s[lane][p*4+2]+s[lane][p*4+3])*.25f;
            means.set(source_x+lane*2+(p&1),p&2?y1:y0,mean);}
        if(vw<4||vh<4){Float4 copy[16];for(int i=0;i<16;++i)copy[i]=s[lane][i];for(uint32_t i=0;i<16;++i)s[lane][i]=copy[bc1::repeat_small_sample(i,vw,vh)];}
    }
    if(srgb)encode_samples_x4<true>(s,out);else encode_samples_x4<false>(s,out);
}
#endif
}
