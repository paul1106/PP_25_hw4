//***********************************************************************************
// 2018.04.01 created by Zexlus1126
// Modified for CUDA GPU acceleration
//
//    CUDA-accelerated Bitcoin Mining
// This is a GPU-accelerated version using CUDA for parallel nonce searching
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>

#include "sha256.h"

////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
} HashBlock;

// Constant memory for block header (80 bytes)
#include "sha256.h"

// circular shift for CPU functions
#define _rotl(v, s) ((v) << (s) | (v) >> (32 - (s)))
#define _rotr(v, s) ((v) >> (s) | (v) << (32 - (s)))
#define _swap(x, y) (((x) ^= (y)), ((y) ^= (x)), ((x) ^= (y)))

// CPU version k array
static const WORD k_cpu[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// CPU SHA-256 functions
void sha256_transform(SHA256 *ctx, const BYTE *msg)
{
    WORD a, b, c, d, e, f, g, h;
    WORD i, j;

    WORD w[64];
    for (i = 0, j = 0; i < 16; ++i, j += 4)
    {
        w[i] = (msg[j] << 24) | (msg[j + 1] << 16) | (msg[j + 2] << 8) | (msg[j + 3]);
    }

    for (i = 16; i < 64; ++i)
    {
        WORD s0 = (_rotr(w[i - 15], 7)) ^ (_rotr(w[i - 15], 18)) ^ (w[i - 15] >> 3);
        WORD s1 = (_rotr(w[i - 2], 17)) ^ (_rotr(w[i - 2], 19)) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    a = ctx->h[0];
    b = ctx->h[1];
    c = ctx->h[2];
    d = ctx->h[3];
    e = ctx->h[4];
    f = ctx->h[5];
    g = ctx->h[6];
    h = ctx->h[7];

    for (i = 0; i < 64; ++i)
    {
        WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
        WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
        WORD ch = (e & f) ^ ((~e) & g);
        WORD maj = (a & b) ^ (a & c) ^ (b & c);
        WORD temp1 = h + S1 + ch + k_cpu[i] + w[i];
        WORD temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx->h[0] += a;
    ctx->h[1] += b;
    ctx->h[2] += c;
    ctx->h[3] += d;
    ctx->h[4] += e;
    ctx->h[5] += f;
    ctx->h[6] += g;
    ctx->h[7] += h;
}

void sha256(SHA256 *ctx, const BYTE *msg, size_t len)
{
    ctx->h[0] = 0x6a09e667;
    ctx->h[1] = 0xbb67ae85;
    ctx->h[2] = 0x3c6ef372;
    ctx->h[3] = 0xa54ff53a;
    ctx->h[4] = 0x510e527f;
    ctx->h[5] = 0x9b05688c;
    ctx->h[6] = 0x1f83d9ab;
    ctx->h[7] = 0x5be0cd19;

    WORD i, j;
    size_t remain = len % 64;
    size_t total_len = len - remain;

    for (i = 0; i < total_len; i += 64)
    {
        sha256_transform(ctx, &msg[i]);
    }

    BYTE m[64] = {};
    for (i = total_len, j = 0; i < len; ++i, ++j)
    {
        m[j] = msg[i];
    }

    m[j++] = 0x80;

    if (j > 56)
    {
        sha256_transform(ctx, m);
        memset(m, 0, sizeof(m));
    }

    unsigned long long L = len * 8;
    m[63] = L;
    m[62] = L >> 8;
    m[61] = L >> 16;
    m[60] = L >> 24;
    m[59] = L >> 32;
    m[58] = L >> 40;
    m[57] = L >> 48;
    m[56] = L >> 56;
    sha256_transform(ctx, m);

    for (i = 0; i < 32; i += 4)
    {
        _swap(ctx->b[i], ctx->b[i + 3]);
        _swap(ctx->b[i + 1], ctx->b[i + 2]);
    }
}

// Constant memory for block header (80 bytes)
__constant__ BYTE dev_block_header[80];

// GPU constant memory k array for SHA-256
__constant__ WORD dev_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// ============== GPU DEVICE FUNCTIONS ==============

__device__ void sha256_transform_device(SHA256 *ctx, const BYTE *msg)
{
    WORD a, b, c, d, e, f, g, h;
    WORD i, j;

    // Create a 64-entry message schedule array w[0..63] of 32-bit words
    WORD w[64];
// Copy chunk into first 16 words w[0..15] of the message schedule array
#pragma unroll
    for (i = 0, j = 0; i < 16; ++i, j += 4)
    {
        w[i] = (msg[j] << 24) | (msg[j + 1] << 16) | (msg[j + 2] << 8) | (msg[j + 3]);
    }

// Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array:
#pragma unroll
    for (i = 16; i < 64; ++i)
    {
        WORD s0 = sig0(w[i - 15]);
        WORD s1 = sig1(w[i - 2]);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    // Initialize working variables to current hash value
    a = ctx->h[0];
    b = ctx->h[1];
    c = ctx->h[2];
    d = ctx->h[3];
    e = ctx->h[4];
    f = ctx->h[5];
    g = ctx->h[6];
    h = ctx->h[7];

// Compress function main loop with unrolling:
#pragma unroll
    for (i = 0; i < 64; ++i)
    {
        WORD S0 = ep0(a);
        WORD S1 = ep1(e);
        WORD ch_val = ch(e, f, g);
        WORD maj_val = maj(a, b, c);
        WORD temp1 = h + S1 + ch_val + dev_k[i] + w[i];
        WORD temp2 = S0 + maj_val;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    // Add the compressed chunk to the current hash value
    ctx->h[0] += a;
    ctx->h[1] += b;
    ctx->h[2] += c;
    ctx->h[3] += d;
    ctx->h[4] += e;
    ctx->h[5] += f;
    ctx->h[6] += g;
    ctx->h[7] += h;
}

__device__ void sha256_device(SHA256 *ctx, const BYTE *msg, size_t len)
{
    // Initialize hash values:
    // (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19):
    ctx->h[0] = 0x6a09e667;
    ctx->h[1] = 0xbb67ae85;
    ctx->h[2] = 0x3c6ef372;
    ctx->h[3] = 0xa54ff53a;
    ctx->h[4] = 0x510e527f;
    ctx->h[5] = 0x9b05688c;
    ctx->h[6] = 0x1f83d9ab;
    ctx->h[7] = 0x5be0cd19;

    WORD i, j;
    size_t remain = len % 64;
    size_t total_len = len - remain;

    // Process the message in successive 512-bit chunks
    // For each chunk:
    for (i = 0; i < total_len; i += 64)
    {
        sha256_transform_device(ctx, &msg[i]);
    }

    // Process remain data
    BYTE m[64] = {};
    for (i = total_len, j = 0; i < len; ++i, ++j)
    {
        m[j] = msg[i];
    }

    // Append a single '1' bit
    m[j++] = 0x80; // 1000 0000

    // Append K '0' bits, where k is the minimum number >= 0 such that L + 1 + K + 64 is a multiple of 512
    if (j > 56)
    {
        sha256_transform_device(ctx, m);
#pragma unroll
        for (int k = 0; k < 64; ++k)
            m[k] = 0;
    }

    // Append L as a 64-bit big-endian integer, making the total post-processed length a multiple of 512 bits
    unsigned long long L = len * 8; // bits
    m[63] = L;
    m[62] = L >> 8;
    m[61] = L >> 16;
    m[60] = L >> 24;
    m[59] = L >> 32;
    m[58] = L >> 40;
    m[57] = L >> 48;
    m[56] = L >> 56;
    sha256_transform_device(ctx, m);

// Produce the final hash value (little-endian to big-endian)
// Swap 1st & 4th, 2nd & 3rd byte for each word
#pragma unroll
    for (i = 0; i < 32; i += 4)
    {
        BYTE temp;
        temp = ctx->b[i];
        ctx->b[i] = ctx->b[i + 3];
        ctx->b[i + 3] = temp;
        temp = ctx->b[i + 1];
        ctx->b[i + 1] = ctx->b[i + 2];
        ctx->b[i + 2] = temp;
    }
}

__device__ void double_sha256_device(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256_device(&tmp, (BYTE *)bytes, len);
    sha256_device(sha256_ctx, (BYTE *)&tmp, sizeof(tmp));
}

// ============== END GPU DEVICE FUNCTIONS ==============

////////////////////////   Utils   ///////////////////////

// convert one hex-codec char to binary
unsigned char decode(unsigned char c)
{
    switch (c)
    {
    case 'a':
        return 0x0a;
    case 'b':
        return 0x0b;
    case 'c':
        return 0x0c;
    case 'd':
        return 0x0d;
    case 'e':
        return 0x0e;
    case 'f':
        return 0x0f;
    case '0' ... '9':
        return c - '0';
    }
    return 0;
}

// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char *out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len / 2 - 1;

    for (; s < string_len; s += 2, --b)
    {
        out[b] = (unsigned char)(decode(in[s]) << 4) + decode(in[s + 1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char *hex, size_t len)
{
    for (int i = 0; i < len; ++i)
    {
        printf("%02x", hex[i]);
    }
}

// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char *hex, size_t len)
{
    for (int i = len - 1; i >= 0; --i)
    {
        printf("%02x", hex[i]);
    }
}

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for (int i = byte_len - 1; i >= 0; --i)
    {
        if (a[i] < b[i])
            return -1;
        else if (a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{

    int i = 0;
    while (i < len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n')
        ;
    str[len - 1] = '\0';
}

////////////////////////   Hash   ///////////////////////

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE *)bytes, len);
    sha256(sha256_ctx, (BYTE *)&tmp, sizeof(tmp));
}

////////////////////   Merkle Root   /////////////////////

// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count; // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count + 1) * 32];
    unsigned char **list = new unsigned char *[total_count + 1];

    // copy each branch to the list
    for (int i = 0; i < total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        // convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count * 32;

    // calculate merkle root
    while (total_count > 1)
    {

        // hash each pair
        int i, j;

        if (total_count % 2 == 1) // odd,
        {
            memcpy(list[total_count], list[total_count - 1], 32);
        }

        for (i = 0, j = 0; i < total_count; i += 2, ++j)
        {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256 *)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}

////////////////////   GPU Mining Kernel   /////////////////////

__global__ void gpu_mine(
    unsigned int *found,
    unsigned int *result_nonce,
    const unsigned char *target,
    unsigned int start_nonce,
    unsigned int total_threads)
{
    // Calculate global thread ID
    unsigned int nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;

    // Early termination check
    if (*found)
        return;

    // Prepare block header with current nonce
    BYTE block[80];

// Copy from constant memory
#pragma unroll
    for (int i = 0; i < 76; ++i)
    {
        block[i] = dev_block_header[i];
    }

    // Add nonce (little-endian)
    block[76] = nonce & 0xff;
    block[77] = (nonce >> 8) & 0xff;
    block[78] = (nonce >> 16) & 0xff;
    block[79] = (nonce >> 24) & 0xff;

    // Perform double SHA-256
    SHA256 hash_result;
    double_sha256_device(&hash_result, block, 80);

    // Compare with target (little-endian comparison)
    bool valid = true;
#pragma unroll
    for (int i = 31; i >= 0; --i)
    {
        if (hash_result.b[i] < target[i])
        {
            break; // hash < target, valid solution
        }
        else if (hash_result.b[i] > target[i])
        {
            valid = false;
            break; // hash > target, invalid
        }
    }

    // If valid solution found, atomically set found flag and store nonce
    if (valid)
    {
        unsigned int old = atomicExch(found, 1);
        if (old == 0)
        {
            *result_nonce = nonce;
        }
    }
}

void solve(FILE *fin, FILE *fout)
{

    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);
    printf("start hashing\n");

    raw_merkle_branch = new char[tx * 65];
    merkle_branch = new char *[tx];
    for (int i = 0; i < tx; ++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    // **** calculate merkle root ****

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    printf("merkle root(little): ");
    print_hex(merkle_root, 32);
    printf("\n");

    printf("merkle root(big):    ");
    print_hex_inverse(merkle_root, 32);
    printf("\n");

    // **** solve block ****
    printf("Block info (big): \n");
    printf("  version:  %s\n", version);
    printf("  pervhash: %s\n", prevhash);
    printf("  merkleroot: ");
    print_hex_inverse(merkle_root, 32);
    printf("\n");
    printf("  nbits:    %s\n", nbits);
    printf("  ntime:    %s\n", ntime);
    printf("  nonce:    ???\n\n");

    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime, ntime, 8);
    block.nonce = 0;

    // ********** calculate target value *********
    // calculate target value from encoded difficulty which is encoded on "nbits"
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};

    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;

    // little-endian
    target_hex[sb] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8 - rb));
    target_hex[sb + 2] = (mant >> (16 - rb));
    target_hex[sb + 3] = (mant >> (24 - rb));

    printf("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    printf("\n");

    // ********** GPU Mining **************

    // Prepare block header (80 bytes) for constant memory
    BYTE host_block_header[80];
    memcpy(host_block_header, &block, 76); // version, prevhash, merkle_root, ntime, nbits
    // nonce will be filled by each thread

    // Copy block header to constant memory
    cudaMemcpyToSymbol(dev_block_header, host_block_header, 76);

    // Allocate device memory for target, found flag, and result
    unsigned char *dev_target;
    unsigned int *dev_found;
    unsigned int *dev_result_nonce;

    cudaMalloc(&dev_target, 32);
    cudaMalloc(&dev_found, sizeof(unsigned int));
    cudaMalloc(&dev_result_nonce, sizeof(unsigned int));

    cudaMemcpy(dev_target, target_hex, 32, cudaMemcpyHostToDevice);

    unsigned int host_found = 0;
    cudaMemcpy(dev_found, &host_found, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    const unsigned int threads_per_block = 256;
    const unsigned int num_blocks = 65536; // Launch many blocks
    const unsigned int total_threads_per_launch = threads_per_block * num_blocks;

    unsigned int start_nonce = 0;
    unsigned int result_nonce = 0;
    bool found = false;

    printf("Starting GPU mining...\n");

    // Launch kernels in batches until solution is found
    while (start_nonce <= 0xffffffff && !found)
    {
        // Reset found flag
        host_found = 0;
        cudaMemcpy(dev_found, &host_found, sizeof(unsigned int), cudaMemcpyHostToDevice);

        // Launch kernel
        gpu_mine<<<num_blocks, threads_per_block>>>(
            dev_found,
            dev_result_nonce,
            dev_target,
            start_nonce,
            total_threads_per_launch);

        cudaDeviceSynchronize();

        // Check if solution was found
        cudaMemcpy(&host_found, dev_found, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        if (host_found)
        {
            cudaMemcpy(&result_nonce, dev_result_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            found = true;
            printf("Found Solution!!\n");
            printf("Nonce: %u (0x%08x)\n", result_nonce, result_nonce);
            break;
        }

        if (start_nonce % 10000000 == 0)
        {
            printf("Searched up to nonce: %u\n", start_nonce);
        }

        // Move to next batch
        start_nonce += total_threads_per_launch;

        // Prevent overflow
        if (start_nonce + total_threads_per_launch < start_nonce)
        {
            break;
        }
    }

    // Verify the solution on CPU
    if (found)
    {
        block.nonce = result_nonce;
        SHA256 sha256_ctx;
        double_sha256(&sha256_ctx, (unsigned char *)&block, sizeof(block));

        printf("hash #%10u (big): ", block.nonce);
        print_hex_inverse(sha256_ctx.b, 32);
        printf("\n\n");

        // little-endian
        printf("hash(little): ");
        print_hex(sha256_ctx.b, 32);
        printf("\n");

        // big-endian
        printf("hash(big):    ");
        print_hex_inverse(sha256_ctx.b, 32);
        printf("\n\n");

        for (int i = 0; i < 4; ++i)
        {
            fprintf(fout, "%02x", ((unsigned char *)&block.nonce)[i]);
        }
        fprintf(fout, "\n");
    }
    else
    {
        printf("No solution found in search space\n");
        fprintf(fout, "00000000\n");
    }

    // Cleanup
    cudaFree(dev_target);
    cudaFree(dev_found);
    cudaFree(dev_result_nonce);

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
        return 1;
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    for (int i = 0; i < totalblock; ++i)
    {
        printf("\n========== Mining Block %d/%d ==========\n", i + 1, totalblock);
        solve(fin, fout);
    }

    fclose(fin);
    fclose(fout);

    return 0;
}
