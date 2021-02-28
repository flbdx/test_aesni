/*
 * Copyright 2021 Florent Bondoux
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "aesni.h"

#include <cstring>

#include <emmintrin.h> // SSE2
#include <immintrin.h> // AVX2
#include <wmmintrin.h> // AESNI

// not sure if this one is standard
#ifndef _MM_SHUFFLE
#   define _MM_SHUFFLE(z, y, x, w) (((z) << 6) | ((y) << 4) | ((x) << 2) | (w))
#endif

#if defined(__clang__)
#   pragma clang attribute push (__attribute__((target("avx2,aes"))), apply_to=function)
#   define INLINE static inline __attribute((always_inline))
#elif defined(__GNUC__)
#   pragma GCC target ("avx2", "aes")
#   define INLINE static inline __attribute((always_inline))
#endif

#ifndef INLINE
#   define INLINE static inline
#endif

namespace AESNI {
    namespace P {
        /*
         * AES-128
         * Create the next round key (k1), using the previous round key (k0)
         */
        template<const int rcon>
        INLINE
        void AES_128_key_exp(__m128i k0, __m128i &k1) {
            __m128i core = _mm_shuffle_epi32(_mm_aeskeygenassist_si128(k0, rcon), _MM_SHUFFLE(3,3,3,3));
            k1 = _mm_xor_si128(k0, _mm_slli_si128(k0, 4));
            k1 = _mm_xor_si128(k1, _mm_slli_si128(k0, 8));
            k1 = _mm_xor_si128(k1, _mm_slli_si128(k0, 12));
            k1 = _mm_xor_si128(k1, core);
        }
        
        /*
         * AES-192
         * Create the next 192 bits of round keys data (k1[64:128] + k2[0:64]) using the previous round keys (k0[0:128] + k1[0:64])
         * AES_192_key_exp_1 and AES_192_key_exp_2 are used alternatively depending on data alignment
         * 
         * k0 [ Prev | Prev | Prev | Prev ]
         * k1 [ Prev | Prev | Next | Next ]
         * k2 [ Next | Next | Next | Next ]
         */
        template<const int rcon>
        INLINE
        void AES_192_key_exp_1(__m128i k0, __m128i &k1, __m128i &k2) {
            __m128i core = _mm_shuffle_epi32(_mm_aeskeygenassist_si128(k1, rcon), _MM_SHUFFLE(1,1,1,1));
            k2 = _mm_xor_si128(k0, _mm_slli_si128(k0, 4));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k0, 8));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k0, 12));
            k2 = _mm_xor_si128(k2, core);
            k2 = _mm_shuffle_epi32(k2, _MM_SHUFFLE(1, 0, 3, 2)); // rotate, bits[64:128] are in position
            k1 = _mm_blend_epi32(k1, k2, 0+0+4+8); // write bits[0:64]
            
            __m128i tmp = _mm_slli_si128(k1, 8);
            tmp = _mm_xor_si128(tmp, _mm_slli_si128(tmp, 4));
            tmp = _mm_xor_si128(tmp, _mm_shuffle_epi32(k2, _MM_SHUFFLE(1,1,1,1)));
            k2 = _mm_blend_epi32(k2, tmp, 0+0+4+8); // blend bits[128:192]
        }
        
        /*
         * AES-192
         * Create the next 192 bits of round keys data (k2[0:128] + k3[0:64]) using the previous round keys (k0[64:128] + k1[0:128])
         * AES_192_key_exp_1 and AES_192_key_exp_2 are used alternatively depending on data alignment
         * 
         * k0 [ XXXX | XXXX | Prev | Prev ]
         * k1 [ Prev | Prev | Prev | Prev ]
         * k2 [ Next | Next | Next | Next ]
         * k3 [ Next | Next | Dirt | Dirt ]
         */
        template<const int rcon>
        INLINE
        void AES_192_key_exp_2(__m128i k0, __m128i k1, __m128i &k2, __m128i &k3) {
            __m128i core = _mm_shuffle_epi32(_mm_aeskeygenassist_si128(k1, rcon), _MM_SHUFFLE(3,3,3,3));
            k2 = _mm_blend_epi32(k0, k1, 1+2+0+0);
            k2 = _mm_shuffle_epi32(k2, _MM_SHUFFLE(1,0,3,2)); // rotate
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k2, 4));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k2, 4));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k2, 4));
            k2 = _mm_xor_si128(k2, core); // write bits[0:128]
            
            k3 = _mm_srli_si128(k1, 8);
            k3 = _mm_xor_si128(k3, _mm_slli_si128(k3, 4));
            k3 = _mm_xor_si128(k3, _mm_shuffle_epi32(k2, _MM_SHUFFLE(3,3,3,3))); // this also override k3[64:128] with dirty data
        }
        
        /*
         * AES-192
         * Create the last 128 bits of round keys data
         * Same as AES_192_key_exp_2 but generate 128 bits
         * 
         * k0 [ XXXX | XXXX | Prev | Prev ]
         * k1 [ Prev | Prev | Prev | Prev ]
         * k2 [ Next | Next | Next | Next ]
         */
        template<const int rcon>
        INLINE
        void AES_192_key_exp_3(__m128i k0, __m128i k1, __m128i &k2) {
            __m128i core = _mm_shuffle_epi32(_mm_aeskeygenassist_si128(k1, rcon), _MM_SHUFFLE(3,3,3,3));
            k2 = _mm_blend_epi32(k0, k1, 1+2+0+0);
            k2 = _mm_shuffle_epi32(k2, _MM_SHUFFLE(1,0,3,2)); // rotate
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k2, 4));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k2, 4));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k2, 4));
            k2 = _mm_xor_si128(k2, core);
        }
        
        /*
         * AES-256
         * Generate the 1st part of the next 256 bits of round keys data using the previous 256 bits
         * Called alternating with AES_256_key_exp_2
         */
        template<const int rcon>
        INLINE
        void AES_256_key_exp_1(__m128i k0, __m128i k1, __m128i &k2) {
            __m128i core = _mm_shuffle_epi32(_mm_aeskeygenassist_si128(k1, rcon), _MM_SHUFFLE(3,3,3,3));
            k2 = _mm_xor_si128(k0, _mm_slli_si128(k0, 4));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k0, 8));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k0, 12));
            k2 = _mm_xor_si128(k2, core);
        }
        
        /*
         * AES-256
         * Generate the 2nd part of next 256 bits of round keys data using the previous 256 bits
         * Called alternating with AES_256_key_exp_1
         */
        INLINE
        void AES_256_key_exp_2(__m128i k0, __m128i k1, __m128i &k2) {
            __m128i sboxed = _mm_shuffle_epi32(_mm_aeskeygenassist_si128(k1, 0), _MM_SHUFFLE(2,2,2,2));
            k2 = _mm_xor_si128(k0, _mm_slli_si128(k0, 4));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k0, 8));
            k2 = _mm_xor_si128(k2, _mm_slli_si128(k0, 12));
            k2 = _mm_xor_si128(k2, sboxed);
        }
        
    }
    
    int AES128_set_encrypt_key(const unsigned char *userKey, AES_KEY_128 *key) {
        if (!userKey || !key) {
            return -1;
        }

        __m128i *rk128 = reinterpret_cast<__m128i *>(key->rd_key);
        
        ::memcpy(rk128, userKey, 16);
        P::AES_128_key_exp<0x01>(rk128[ 0], rk128[ 1]);
        P::AES_128_key_exp<0x02>(rk128[ 1], rk128[ 2]);
        P::AES_128_key_exp<0x04>(rk128[ 2], rk128[ 3]);
        P::AES_128_key_exp<0x08>(rk128[ 3], rk128[ 4]);
        P::AES_128_key_exp<0x10>(rk128[ 4], rk128[ 5]);
        P::AES_128_key_exp<0x20>(rk128[ 5], rk128[ 6]);
        P::AES_128_key_exp<0x40>(rk128[ 6], rk128[ 7]);
        P::AES_128_key_exp<0x80>(rk128[ 7], rk128[ 8]);
        P::AES_128_key_exp<0x1B>(rk128[ 8], rk128[ 9]);
        P::AES_128_key_exp<0x36>(rk128[ 9], rk128[10]);
        
        return 0;
    }
    
    int AES128_set_decrypt_key(const unsigned char *userKey, AES_KEY_128 *key) {
        if (!userKey || !key) {
            return -1;
        }

        __m128i *rk128 = reinterpret_cast<__m128i *>(key->rd_key);
        
        ::memcpy(&(rk128[10]), userKey, 16);
        P::AES_128_key_exp<0x01>(rk128[10], rk128[ 9]);
        P::AES_128_key_exp<0x02>(rk128[ 9], rk128[ 8]);
        rk128[ 9] = _mm_aesimc_si128(rk128[9]);
        P::AES_128_key_exp<0x04>(rk128[ 8], rk128[ 7]);
        rk128[ 8] = _mm_aesimc_si128(rk128[8]);
        P::AES_128_key_exp<0x08>(rk128[ 7], rk128[ 6]);
        rk128[ 7] = _mm_aesimc_si128(rk128[7]);
        P::AES_128_key_exp<0x10>(rk128[ 6], rk128[ 5]);
        rk128[ 6] = _mm_aesimc_si128(rk128[6]);
        P::AES_128_key_exp<0x20>(rk128[ 5], rk128[ 4]);
        rk128[ 5] = _mm_aesimc_si128(rk128[5]);
        P::AES_128_key_exp<0x40>(rk128[ 4], rk128[ 3]);
        rk128[ 4] = _mm_aesimc_si128(rk128[4]);
        P::AES_128_key_exp<0x80>(rk128[ 3], rk128[ 2]);
        rk128[ 3] = _mm_aesimc_si128(rk128[3]);
        P::AES_128_key_exp<0x1B>(rk128[ 2], rk128[ 1]);
        rk128[ 2] = _mm_aesimc_si128(rk128[2]);
        P::AES_128_key_exp<0x36>(rk128[ 1], rk128[ 0]);
        rk128[ 1] = _mm_aesimc_si128(rk128[1]);
        
        return 0;
    }
    
    int AES192_set_encrypt_key(const unsigned char *userKey, AES_KEY_192 *key) {
        if (!userKey || !key) {
            return -1;
        }

        __m128i *rk128 = reinterpret_cast<__m128i *>(key->rd_key);
        
        ::memcpy(rk128, userKey, 24);
        P::AES_192_key_exp_1<0x01>(rk128[ 0], rk128[ 1], rk128[ 2]);
        P::AES_192_key_exp_2<0x02>(rk128[ 1], rk128[ 2], rk128[ 3], rk128[ 4]);
        P::AES_192_key_exp_1<0x04>(rk128[ 3], rk128[ 4], rk128[ 5]);
        P::AES_192_key_exp_2<0x08>(rk128[ 4], rk128[ 5], rk128[ 6], rk128[ 7]);
        P::AES_192_key_exp_1<0x10>(rk128[ 6], rk128[ 7], rk128[ 8]);
        P::AES_192_key_exp_2<0x20>(rk128[ 7], rk128[ 8], rk128[ 9], rk128[10]);
        P::AES_192_key_exp_1<0x40>(rk128[ 9], rk128[10], rk128[11]);
        P::AES_192_key_exp_3<0x80>(rk128[10], rk128[11], rk128[12]);
        
        return 0;
    }
    
    int AES192_set_decrypt_key(const unsigned char *userKey, AES_KEY_192 *key) {
        if (!userKey || !key) {
            return -1;
        }

        __m128i *rk128 = reinterpret_cast<__m128i *>(key->rd_key);
        
        ::memcpy(&(rk128[12]), userKey, 16);
        ::memcpy(&(rk128[11]), userKey + 16, 8);
        P::AES_192_key_exp_1<0x01>(rk128[12], rk128[11], rk128[10]);
        P::AES_192_key_exp_2<0x02>(rk128[11], rk128[10], rk128[ 9], rk128[ 8]);
        rk128[11] = _mm_aesimc_si128(rk128[11]);
        rk128[10] = _mm_aesimc_si128(rk128[10]);
        P::AES_192_key_exp_1<0x04>(rk128[ 9], rk128[ 8], rk128[ 7]);
        rk128[ 9] = _mm_aesimc_si128(rk128[ 9]);
        P::AES_192_key_exp_2<0x08>(rk128[ 8], rk128[ 7], rk128[ 6], rk128[ 5]);
        rk128[ 8] = _mm_aesimc_si128(rk128[ 8]);
        rk128[ 7] = _mm_aesimc_si128(rk128[ 7]);
        P::AES_192_key_exp_1<0x10>(rk128[ 6], rk128[ 5], rk128[ 4]);
        rk128[ 6] = _mm_aesimc_si128(rk128[ 6]);
        P::AES_192_key_exp_2<0x20>(rk128[ 5], rk128[ 4], rk128[ 3], rk128[ 2]);
        rk128[ 5] = _mm_aesimc_si128(rk128[ 5]);
        rk128[ 4] = _mm_aesimc_si128(rk128[ 4]);
        P::AES_192_key_exp_1<0x40>(rk128[ 3], rk128[ 2], rk128[ 1]);
        rk128[ 3] = _mm_aesimc_si128(rk128[ 3]);
        P::AES_192_key_exp_3<0x80>(rk128[ 2], rk128[ 1], rk128[ 0]);
        rk128[ 2] = _mm_aesimc_si128(rk128[ 2]);
        rk128[ 1] = _mm_aesimc_si128(rk128[ 1]);
        
        return 0;
    }
    
    int AES256_set_encrypt_key(const unsigned char *userKey, AES_KEY_256 *key) {
        if (!userKey || !key) {
            return -1;
        }

        __m128i *rk128 = reinterpret_cast<__m128i *>(key->rd_key);
        
        ::memcpy(rk128, userKey, 32);
        P::AES_256_key_exp_1<0x01>(rk128[ 0], rk128[ 1], rk128[ 2]);
        P::AES_256_key_exp_2      (rk128[ 1], rk128[ 2], rk128[ 3]);
        P::AES_256_key_exp_1<0x02>(rk128[ 2], rk128[ 3], rk128[ 4]);
        P::AES_256_key_exp_2      (rk128[ 3], rk128[ 4], rk128[ 5]);
        P::AES_256_key_exp_1<0x04>(rk128[ 4], rk128[ 5], rk128[ 6]);
        P::AES_256_key_exp_2      (rk128[ 5], rk128[ 6], rk128[ 7]);
        P::AES_256_key_exp_1<0x08>(rk128[ 6], rk128[ 7], rk128[ 8]);
        P::AES_256_key_exp_2      (rk128[ 7], rk128[ 8], rk128[ 9]);
        P::AES_256_key_exp_1<0x10>(rk128[ 8], rk128[ 9], rk128[10]);
        P::AES_256_key_exp_2      (rk128[ 9], rk128[10], rk128[11]);
        P::AES_256_key_exp_1<0x20>(rk128[10], rk128[11], rk128[12]);
        P::AES_256_key_exp_2      (rk128[11], rk128[12], rk128[13]);
        P::AES_256_key_exp_1<0x40>(rk128[12], rk128[13], rk128[14]);
        
        return 0;
    }
    
    int AES256_set_decrypt_key(const unsigned char *userKey, AES_KEY_256 *key) {
        if (!userKey || !key) {
            return -1;
        }

        __m128i *rk128 = reinterpret_cast<__m128i *>(key->rd_key);
        
        ::memcpy(&(rk128[14]), userKey, 16);
        ::memcpy(&(rk128[13]), userKey + 16, 16);
        P::AES_256_key_exp_1<0x01>(rk128[14], rk128[13], rk128[12]);
        P::AES_256_key_exp_2      (rk128[13], rk128[12], rk128[11]);
        rk128[13] = _mm_aesimc_si128(rk128[13]);
        P::AES_256_key_exp_1<0x02>(rk128[12], rk128[11], rk128[10]);
        rk128[12] = _mm_aesimc_si128(rk128[12]);
        P::AES_256_key_exp_2      (rk128[11], rk128[10], rk128[ 9]);
        rk128[11] = _mm_aesimc_si128(rk128[11]);
        P::AES_256_key_exp_1<0x04>(rk128[10], rk128[ 9], rk128[ 8]);
        rk128[10] = _mm_aesimc_si128(rk128[10]);
        P::AES_256_key_exp_2      (rk128[ 9], rk128[ 8], rk128[ 7]);
        rk128[ 9] = _mm_aesimc_si128(rk128[ 9]);
        P::AES_256_key_exp_1<0x08>(rk128[ 8], rk128[ 7], rk128[ 6]);
        rk128[ 8] = _mm_aesimc_si128(rk128[ 8]);
        P::AES_256_key_exp_2      (rk128[ 7], rk128[ 6], rk128[ 5]);
        rk128[ 7] = _mm_aesimc_si128(rk128[ 7]);
        P::AES_256_key_exp_1<0x10>(rk128[ 6], rk128[ 5], rk128[ 4]);
        rk128[ 6] = _mm_aesimc_si128(rk128[ 6]);
        P::AES_256_key_exp_2      (rk128[ 5], rk128[ 4], rk128[ 3]);
        rk128[ 5] = _mm_aesimc_si128(rk128[ 5]);
        P::AES_256_key_exp_1<0x20>(rk128[ 4], rk128[ 3], rk128[ 2]);
        rk128[ 4] = _mm_aesimc_si128(rk128[ 4]);
        P::AES_256_key_exp_2      (rk128[ 3], rk128[ 2], rk128[ 1]);
        rk128[ 3] = _mm_aesimc_si128(rk128[ 3]);
        P::AES_256_key_exp_1<0x40>(rk128[ 2], rk128[ 1], rk128[ 0]);
        rk128[ 2] = _mm_aesimc_si128(rk128[ 2]);
        rk128[ 1] = _mm_aesimc_si128(rk128[ 1]);
        
        return 0;
    }
    
    int AES_set_encrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key) {
        if (bits == 128) {
            if (AES128_set_encrypt_key(userKey, reinterpret_cast<AES_KEY_128 *>(key)) == 0) {
                key->rounds = 10;
                return 0;
            }
        }
        else if (bits == 192) {
            if (AES192_set_encrypt_key(userKey, reinterpret_cast<AES_KEY_192 *>(key)) == 0) {
                key->rounds = 12;
                return 0;
            }
        }
        else if (bits == 256) {
            if (AES256_set_encrypt_key(userKey, reinterpret_cast<AES_KEY_256 *>(key)) == 0) {
                key->rounds = 14;
                return 0;
            }
        }
        
        return -2;
    }
    
    int AES_set_decrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key) {
        if (bits == 128) {
            if (AES128_set_decrypt_key(userKey, reinterpret_cast<AES_KEY_128 *>(key)) == 0) {
                key->rounds = 10;
                return 0;
            }
        }
        else if (bits == 192) {
            if (AES192_set_decrypt_key(userKey, reinterpret_cast<AES_KEY_192 *>(key)) == 0) {
                key->rounds = 12;
                return 0;
            }
        }
        else if (bits == 256) {
            if (AES256_set_decrypt_key(userKey, reinterpret_cast<AES_KEY_256 *>(key)) == 0) {
                key->rounds = 14;
                return 0;
            }
        }
        
        return -2;
    }
    
    void AES128_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY_128 *key) {
        __m128i m = _mm_loadu_si128((__m128i *) in);
        const __m128i *rk128 = reinterpret_cast<const __m128i *>(key->rd_key);

        m = _mm_xor_si128(m, rk128[ 0]);
        m = _mm_aesenc_si128(m, rk128[ 1]);
        m = _mm_aesenc_si128(m, rk128[ 2]);
        m = _mm_aesenc_si128(m, rk128[ 3]);
        m = _mm_aesenc_si128(m, rk128[ 4]);
        m = _mm_aesenc_si128(m, rk128[ 5]);
        m = _mm_aesenc_si128(m, rk128[ 6]);
        m = _mm_aesenc_si128(m, rk128[ 7]);
        m = _mm_aesenc_si128(m, rk128[ 8]);
        m = _mm_aesenc_si128(m, rk128[ 9]);
        m = _mm_aesenclast_si128(m, rk128[10]);

        _mm_storeu_si128((__m128i *) out, m);
    }

    void AES192_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY_192 *key) {
        __m128i m = _mm_loadu_si128((__m128i *) in);
        const __m128i *rk128 = reinterpret_cast<const __m128i *>(key->rd_key);

        m = _mm_xor_si128(m, rk128[ 0]);
        m = _mm_aesenc_si128(m, rk128[ 1]);
        m = _mm_aesenc_si128(m, rk128[ 2]);
        m = _mm_aesenc_si128(m, rk128[ 3]);
        m = _mm_aesenc_si128(m, rk128[ 4]);
        m = _mm_aesenc_si128(m, rk128[ 5]);
        m = _mm_aesenc_si128(m, rk128[ 6]);
        m = _mm_aesenc_si128(m, rk128[ 7]);
        m = _mm_aesenc_si128(m, rk128[ 8]);
        m = _mm_aesenc_si128(m, rk128[ 9]);
        m = _mm_aesenc_si128(m, rk128[10]);
        m = _mm_aesenc_si128(m, rk128[11]);
        m = _mm_aesenclast_si128(m, rk128[12]);

        _mm_storeu_si128((__m128i *) out, m);
    }
    
    void AES256_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY_256 *key) {
        __m128i m = _mm_loadu_si128((__m128i *) in);
        const __m128i *rk128 = reinterpret_cast<const __m128i *>(key->rd_key);

        m = _mm_xor_si128(m, rk128[ 0]);
        m = _mm_aesenc_si128(m, rk128[ 1]);
        m = _mm_aesenc_si128(m, rk128[ 2]);
        m = _mm_aesenc_si128(m, rk128[ 3]);
        m = _mm_aesenc_si128(m, rk128[ 4]);
        m = _mm_aesenc_si128(m, rk128[ 5]);
        m = _mm_aesenc_si128(m, rk128[ 6]);
        m = _mm_aesenc_si128(m, rk128[ 7]);
        m = _mm_aesenc_si128(m, rk128[ 8]);
        m = _mm_aesenc_si128(m, rk128[ 9]);
        m = _mm_aesenc_si128(m, rk128[10]);
        m = _mm_aesenc_si128(m, rk128[11]);
        m = _mm_aesenc_si128(m, rk128[12]);
        m = _mm_aesenc_si128(m, rk128[13]);
        m = _mm_aesenclast_si128(m, rk128[14]);

        _mm_storeu_si128((__m128i *) out, m);
    }
    
    void AES_encrypt(const unsigned char *in, unsigned char *out,
                 const AES_KEY *key) {
        __m128i m = _mm_loadu_si128((__m128i *) in);
        const __m128i *rk128 = reinterpret_cast<const __m128i *>(key->rd_key);

        m = _mm_xor_si128(m, rk128[ 0]);
        m = _mm_aesenc_si128(m, rk128[ 1]);
        m = _mm_aesenc_si128(m, rk128[ 2]);
        m = _mm_aesenc_si128(m, rk128[ 3]);
        m = _mm_aesenc_si128(m, rk128[ 4]);
        m = _mm_aesenc_si128(m, rk128[ 5]);
        m = _mm_aesenc_si128(m, rk128[ 6]);
        m = _mm_aesenc_si128(m, rk128[ 7]);
        m = _mm_aesenc_si128(m, rk128[ 8]);
        m = _mm_aesenc_si128(m, rk128[ 9]);
        if (key->rounds == 10) {
            m = _mm_aesenclast_si128(m, rk128[10]);
        }
        else if (key->rounds == 12) {
            m = _mm_aesenc_si128(m, rk128[10]);
            m = _mm_aesenc_si128(m, rk128[11]);
            m = _mm_aesenclast_si128(m, rk128[12]);
        }
        else {
            m = _mm_aesenc_si128(m, rk128[10]);
            m = _mm_aesenc_si128(m, rk128[11]);
            m = _mm_aesenc_si128(m, rk128[12]);
            m = _mm_aesenc_si128(m, rk128[13]);
            m = _mm_aesenclast_si128(m, rk128[14]);
        }

        _mm_storeu_si128((__m128i *) out, m);
    }
    
    void AES128_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY_128 *key) {
        __m128i m = _mm_loadu_si128((__m128i *) in);
        const __m128i *rk128 = reinterpret_cast<const __m128i *>(key->rd_key);

        m = _mm_xor_si128(m, rk128[ 0]);
        m = _mm_aesdec_si128(m, rk128[ 1]);
        m = _mm_aesdec_si128(m, rk128[ 2]);
        m = _mm_aesdec_si128(m, rk128[ 3]);
        m = _mm_aesdec_si128(m, rk128[ 4]);
        m = _mm_aesdec_si128(m, rk128[ 5]);
        m = _mm_aesdec_si128(m, rk128[ 6]);
        m = _mm_aesdec_si128(m, rk128[ 7]);
        m = _mm_aesdec_si128(m, rk128[ 8]);
        m = _mm_aesdec_si128(m, rk128[ 9]);
        m = _mm_aesdeclast_si128(m, rk128[10]);

        _mm_storeu_si128((__m128i *) out, m);
    }
    
    void AES192_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY_192 *key) {
        __m128i m = _mm_loadu_si128((__m128i *) in);
        const __m128i *rk128 = reinterpret_cast<const __m128i *>(key->rd_key);

        m = _mm_xor_si128(m, rk128[ 0]);
        m = _mm_aesdec_si128(m, rk128[ 1]);
        m = _mm_aesdec_si128(m, rk128[ 2]);
        m = _mm_aesdec_si128(m, rk128[ 3]);
        m = _mm_aesdec_si128(m, rk128[ 4]);
        m = _mm_aesdec_si128(m, rk128[ 5]);
        m = _mm_aesdec_si128(m, rk128[ 6]);
        m = _mm_aesdec_si128(m, rk128[ 7]);
        m = _mm_aesdec_si128(m, rk128[ 8]);
        m = _mm_aesdec_si128(m, rk128[ 9]);
        m = _mm_aesdec_si128(m, rk128[10]);
        m = _mm_aesdec_si128(m, rk128[11]);
        m = _mm_aesdeclast_si128(m, rk128[12]);

        _mm_storeu_si128((__m128i *) out, m);
    }
    
    void AES256_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY_256 *key) {
        __m128i m = _mm_loadu_si128((__m128i *) in);
        const __m128i *rk128 = reinterpret_cast<const __m128i *>(key->rd_key);

        m = _mm_xor_si128(m, rk128[ 0]);
        m = _mm_aesdec_si128(m, rk128[ 1]);
        m = _mm_aesdec_si128(m, rk128[ 2]);
        m = _mm_aesdec_si128(m, rk128[ 3]);
        m = _mm_aesdec_si128(m, rk128[ 4]);
        m = _mm_aesdec_si128(m, rk128[ 5]);
        m = _mm_aesdec_si128(m, rk128[ 6]);
        m = _mm_aesdec_si128(m, rk128[ 7]);
        m = _mm_aesdec_si128(m, rk128[ 8]);
        m = _mm_aesdec_si128(m, rk128[ 9]);
        m = _mm_aesdec_si128(m, rk128[10]);
        m = _mm_aesdec_si128(m, rk128[11]);
        m = _mm_aesdec_si128(m, rk128[12]);
        m = _mm_aesdec_si128(m, rk128[13]);
        m = _mm_aesdeclast_si128(m, rk128[14]);

        _mm_storeu_si128((__m128i *) out, m);
    }
    
    void AES_decrypt(const unsigned char *in, unsigned char *out,
                 const AES_KEY *key) {
        __m128i m = _mm_loadu_si128((__m128i *) in);
        const __m128i *rk128 = reinterpret_cast<const __m128i *>(key->rd_key);

        m = _mm_xor_si128(m, rk128[ 0]);
        m = _mm_aesdec_si128(m, rk128[ 1]);
        m = _mm_aesdec_si128(m, rk128[ 2]);
        m = _mm_aesdec_si128(m, rk128[ 3]);
        m = _mm_aesdec_si128(m, rk128[ 4]);
        m = _mm_aesdec_si128(m, rk128[ 5]);
        m = _mm_aesdec_si128(m, rk128[ 6]);
        m = _mm_aesdec_si128(m, rk128[ 7]);
        m = _mm_aesdec_si128(m, rk128[ 8]);
        m = _mm_aesdec_si128(m, rk128[ 9]);
        if (key->rounds == 10) {
            m = _mm_aesdeclast_si128(m, rk128[10]);
        }
        else if (key->rounds == 12) {
            m = _mm_aesdec_si128(m, rk128[10]);
            m = _mm_aesdec_si128(m, rk128[11]);
            m = _mm_aesdeclast_si128(m, rk128[12]);
        }
        else {
            m = _mm_aesdec_si128(m, rk128[10]);
            m = _mm_aesdec_si128(m, rk128[11]);
            m = _mm_aesdec_si128(m, rk128[12]);
            m = _mm_aesdec_si128(m, rk128[13]);
            m = _mm_aesdeclast_si128(m, rk128[14]);
        }

        _mm_storeu_si128((__m128i *) out, m);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
