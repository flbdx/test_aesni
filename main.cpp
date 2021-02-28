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
#include <cstdio>

#include <openssl/aes.h>
#include <openssl/rand.h>

template<const int bits, typename KEY,
    int set_encrypt_key(const unsigned char *, KEY *),
    int set_decrypt_key(const unsigned char *, KEY *),
    void encrypt(const unsigned char *, unsigned char *, const KEY *),
    void decrypt(const unsigned char *, unsigned char *, const KEY *)>
bool rand_test() {
    uint8_t key[32];
    uint8_t in[16];
    uint8_t out_ref[16], out_tst[16];
    RAND_bytes(key, 32);
    
    ::AES_KEY ref_enc_key;
    ::AES_set_encrypt_key(key, bits, &ref_enc_key);
    ::AES_encrypt(in, out_ref, &ref_enc_key);
    
    KEY enc_key, dec_key;
    set_encrypt_key(key, &enc_key);
    set_decrypt_key(key, &dec_key);
    
    encrypt(in, out_tst, &enc_key);
    if (::memcmp(out_ref, out_tst, 16) != 0) {
        printf("error: AESNI::AES%d_encrypt\n", bits);
        return false;
    }
    
    decrypt(out_tst, out_tst, &dec_key);
    if (::memcmp(in, out_tst, 16) != 0) {
        printf("error: AESNI::AES%d_decrypt\n", bits);
        return false;
    }
    
    AESNI::AES_KEY enc_key_gen, dec_key_gen;
    AESNI::AES_set_encrypt_key(key, bits, &enc_key_gen);
    AESNI::AES_set_decrypt_key(key, bits, &dec_key_gen);
    
    AESNI::AES_encrypt(in, out_tst, &enc_key_gen);
    if (::memcmp(out_ref, out_tst, 16) != 0) {
        printf("error: AESNI::AES_encrypt (%d)\n", bits);
        return false;
    }
    
    AESNI::AES_decrypt(out_tst, out_tst, &dec_key_gen);
    if (::memcmp(in, out_tst, 16) != 0) {
        printf("error: AESNI::AES_encrypt (%d)\n", bits);
        return false;
    }
    
    return true;
}



int main() {
    const int n_tests = 200000;
    puts("AES-128...");
    for (int i = 0; i < n_tests; i++) {
        if (!rand_test<128, AESNI::AES_KEY_128, AESNI::AES128_set_encrypt_key, AESNI::AES128_set_decrypt_key, AESNI::AES128_encrypt, AESNI::AES128_decrypt>()) {
            break;
        }
    }
    puts("AES-192...");
    for (int i = 0; i < n_tests; i++) {
        if (!rand_test<192, AESNI::AES_KEY_192, AESNI::AES192_set_encrypt_key, AESNI::AES192_set_decrypt_key, AESNI::AES192_encrypt, AESNI::AES192_decrypt>()) {
            break;
        }
    }
    puts("AES-256...");
    for (int i = 0; i < n_tests; i++) {
        if (!rand_test<256, AESNI::AES_KEY_256, AESNI::AES256_set_encrypt_key, AESNI::AES256_set_decrypt_key, AESNI::AES256_encrypt, AESNI::AES256_decrypt>()) {
            break;
        }
    }
    
    
    return 0;
}
