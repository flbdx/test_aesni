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
#include <chrono>

#include <openssl/aes.h>
#include <openssl/rand.h>
#include <openssl/evp.h>

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

template<const int bits_, typename Key_,
    int set_encrypt_key_(const unsigned char *, Key_ *),
    int set_decrypt_key_(const unsigned char *, Key_ *),
    void encrypt_(const unsigned char *, unsigned char *, const Key_ *),
    void decrypt_(const unsigned char *, unsigned char *, const Key_ *)>
struct AESNIDescriptor {
    typedef Key_ Key;
    
    static inline int bits() {
        return bits_;
    }
    
    static inline int set_encrypt_key(const unsigned char *userKey, Key *key) {
        return set_encrypt_key_(userKey, key);
    }
    static inline int set_decrypt_key(const unsigned char *userKey, Key *key) {
        return set_decrypt_key_(userKey, key);
    }
    static inline void encrypt(const unsigned char *in, unsigned char *out, const Key *key) {
        encrypt_(in, out, key);
    }
    static inline void decrypt(const unsigned char *in, unsigned char *out, const Key *key) {
        decrypt_(in, out, key);
    }
};

template<const int bits_>
struct OpenSSLDesc {
    typedef ::AES_KEY Key;
    
    static inline int bits() {
        return bits_;
    }
    
    static inline int set_encrypt_key(const unsigned char *userKey, Key *key) {
        return ::AES_set_encrypt_key(userKey, bits_, key);
    }
    static inline int set_decrypt_key(const unsigned char *userKey, Key *key) {
        return ::AES_set_decrypt_key(userKey, bits_, key);
    }
    static inline void encrypt(const unsigned char *in, unsigned char *out, const Key *key) {
        ::AES_encrypt(in, out, key);
    }
    static inline void decrypt(const unsigned char *in, unsigned char *out, const Key *key) {
        ::AES_decrypt(in, out, key);
    }
};

typedef AESNIDescriptor<128, AESNI::AES_KEY_128, AESNI::AES128_set_encrypt_key, AESNI::AES128_set_decrypt_key, AESNI::AES128_encrypt, AESNI::AES128_decrypt> AESNIDesc128;
typedef AESNIDescriptor<192, AESNI::AES_KEY_192, AESNI::AES192_set_encrypt_key, AESNI::AES192_set_decrypt_key, AESNI::AES192_encrypt, AESNI::AES192_decrypt> AESNIDesc192;
typedef AESNIDescriptor<256, AESNI::AES_KEY_256, AESNI::AES256_set_encrypt_key, AESNI::AES256_set_decrypt_key, AESNI::AES256_encrypt, AESNI::AES256_decrypt> AESNIDesc256;
typedef OpenSSLDesc<128> OpenSSLDesc128;
typedef OpenSSLDesc<192> OpenSSLDesc192;
typedef OpenSSLDesc<256> OpenSSLDesc256;

template<typename Desc, const unsigned int n_keys = 256, const unsigned long n_runs = 100000>
class SpeedTest
{
    unsigned char m_userKeys[n_keys * 32];
    typename Desc::Key m_keys[n_keys];
    
public:
    
    SpeedTest() : m_userKeys(), m_keys() {
        RAND_bytes(m_userKeys, sizeof(m_userKeys));
    }
    
    void run_set_encrypt_key() {
        auto begin = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < n_runs; r++) {
            for (int n = 0; n < n_keys; n++) {
                Desc::set_encrypt_key(m_userKeys + n  * 32, m_keys + n);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> d = end - begin;
        printf("    %0.3f key setup / ms\n", (n_keys * n_runs) / d.count());
    }
    
    void run_set_decrypt_key() {
        auto begin = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < n_runs; r++) {
            for (int n = 0; n < n_keys; n++) {
                Desc::set_decrypt_key(m_userKeys + n  * 32, m_keys + n);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> d = end - begin;
        printf("    %0.3f key setup / ms\n", (n_keys * n_runs) / d.count());
    }
};

template<const int bits, const unsigned int n_keys = 256, const unsigned long n_runs = 100000>
class SpeedTestEVP
{
    unsigned char m_userKeys[n_keys * 32];
    EVP_CIPHER_CTX *m_ctx[n_keys];
    const EVP_CIPHER *m_cipher;
    
public:
    
    SpeedTestEVP() : m_userKeys(), m_ctx(), m_cipher(NULL) {
        RAND_bytes(m_userKeys, sizeof(m_userKeys));
        for (int i = 0; i < n_keys; i++) {
            m_ctx[i] = EVP_CIPHER_CTX_new();
        }
        if (bits == 128) {
            m_cipher = EVP_aes_128_ecb();
        }
        else if (bits == 192) {
            m_cipher = EVP_aes_192_ecb();
        }
        else {
            m_cipher = EVP_aes_256_ecb();
        }
    }
    
    ~SpeedTestEVP() {
        for (int i = 0; i < n_keys; i++) {
            EVP_CIPHER_CTX_free(m_ctx[i]);
        }
    }
    
    void run_set_encrypt_key() {
        auto begin = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < n_runs; r++) {
            for (int n = 0; n < n_keys; n++) {
                EVP_EncryptInit_ex(m_ctx[n], m_cipher, NULL, m_userKeys + n * 32, NULL);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> d = end - begin;
        printf("    %0.3f key setup / ms\n", (n_keys * n_runs) / d.count());
    }
    
    void run_set_decrypt_key() {
        auto begin = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < n_runs; r++) {
            for (int n = 0; n < n_keys; n++) {
                EVP_DecryptInit_ex(m_ctx[n], m_cipher, NULL, m_userKeys + n * 32, NULL);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> d = end - begin;
        printf("    %0.3f key setup / ms\n", (n_keys * n_runs) / d.count());
    }
};

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
    
    puts("AESNI set_encrypt_key 128");
    SpeedTest<AESNIDesc128>().run_set_encrypt_key();
    puts("AESNI set_decrypt_key 128");
    SpeedTest<AESNIDesc128>().run_set_decrypt_key();
    puts("AESNI set_encrypt_key 192");
    SpeedTest<AESNIDesc192>().run_set_encrypt_key();
    puts("AESNI set_decrypt_key 192");
    SpeedTest<AESNIDesc192>().run_set_decrypt_key();
    puts("AESNI set_encrypt_key 256");
    SpeedTest<AESNIDesc256>().run_set_encrypt_key();
    puts("AESNI set_decrypt_key 256");
    SpeedTest<AESNIDesc256>().run_set_decrypt_key();
    
    puts("OpenSSL set_encrypt_key 128");
    SpeedTest<OpenSSLDesc128>().run_set_encrypt_key();
    puts("OpenSSL set_decrypt_key 128");
    SpeedTest<OpenSSLDesc128>().run_set_decrypt_key();
    puts("OpenSSL set_encrypt_key 192");
    SpeedTest<OpenSSLDesc192>().run_set_encrypt_key();
    puts("OpenSSL set_decrypt_key 192");
    SpeedTest<OpenSSLDesc192>().run_set_decrypt_key();
    puts("OpenSSL set_encrypt_key 256");
    SpeedTest<OpenSSLDesc256>().run_set_encrypt_key();
    puts("OpenSSL set_decrypt_key 256");
    SpeedTest<OpenSSLDesc256>().run_set_decrypt_key();
    
    puts("OpenSSL EVP_EncryptInit_ex 128");
    SpeedTestEVP<128>().run_set_encrypt_key();
    puts("OpenSSL EVP_DecryptInit_ex 128");
    SpeedTestEVP<128>().run_set_decrypt_key();
    puts("OpenSSL EVP_EncryptInit_ex 192");
    SpeedTestEVP<192>().run_set_encrypt_key();
    puts("OpenSSL EVP_DecryptInit_ex 192");
    SpeedTestEVP<192>().run_set_decrypt_key();
    puts("OpenSSL EVP_EncryptInit_ex 256");
    SpeedTestEVP<256>().run_set_encrypt_key();
    puts("OpenSSL EVP_DecryptInit_ex 256");
    SpeedTestEVP<256>().run_set_decrypt_key();
    
    return 0;
}
