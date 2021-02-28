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

#pragma once

#include <cstdint>

namespace AESNI {
    struct AES_KEY_128 {
        __attribute__((aligned(16))) uint32_t rd_key[4 * (10 + 1)];
    };
    
    struct AES_KEY_192 {
        __attribute__((aligned(16))) uint32_t rd_key[4 * (12 + 1)];
    };
    
    struct AES_KEY_256 {
        __attribute__((aligned(16))) uint32_t rd_key[4 * (14 + 1)];
    };
    
    struct AES_KEY {
        __attribute__((aligned(16))) uint32_t rd_key[4 * (14 + 1)];
        int rounds;
    };
    
    int AES128_set_encrypt_key(const unsigned char *userKey, AES_KEY_128 *key);
    int AES128_set_decrypt_key(const unsigned char *userKey, AES_KEY_128 *key);
    void AES128_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY_128 *key);
    void AES128_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY_128 *key);
    
    int AES192_set_encrypt_key(const unsigned char *userKey, AES_KEY_192 *key);
    int AES192_set_decrypt_key(const unsigned char *userKey, AES_KEY_192 *key);
    void AES192_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY_192 *key);
    void AES192_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY_192 *key);
    
    int AES256_set_encrypt_key(const unsigned char *userKey, AES_KEY_256 *key);
    int AES256_set_decrypt_key(const unsigned char *userKey, AES_KEY_256 *key);
    void AES256_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY_256 *key);
    void AES256_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY_256 *key);
    
    int AES_set_encrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key);
    int AES_set_decrypt_key(const unsigned char *userKey, const int bits, AES_KEY *key);
    void AES_encrypt(const unsigned char *in, unsigned char *out, const AES_KEY *key);
    void AES_decrypt(const unsigned char *in, unsigned char *out, const AES_KEY *key);
}
