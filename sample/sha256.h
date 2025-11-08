#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif //__cplusplus

	//--------------- DATA TYPES --------------
	typedef unsigned int WORD;
	typedef unsigned char BYTE;

	typedef union _sha256_ctx
	{
		WORD h[8];
		BYTE b[32];
	} SHA256;

	//----------- CPU FUNCTION DECLARATION --------
	void sha256_transform(SHA256 *ctx, const BYTE *msg);
	void sha256(SHA256 *ctx, const BYTE *msg, size_t len);

	//----------- GPU DEVICE FUNCTIONS ---------
	// Inline device functions for bit operations
	__device__ __forceinline__ WORD rotr(WORD x, WORD n)
	{
		return (x >> n) | (x << (32 - n));
	}

	__device__ __forceinline__ WORD ch(WORD x, WORD y, WORD z)
	{
		return (x & y) ^ (~x & z);
	}

	__device__ __forceinline__ WORD maj(WORD x, WORD y, WORD z)
	{
		return (x & y) ^ (x & z) ^ (y & z);
	}

	__device__ __forceinline__ WORD ep0(WORD x)
	{
		return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
	}

	__device__ __forceinline__ WORD ep1(WORD x)
	{
		return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
	}

	__device__ __forceinline__ WORD sig0(WORD x)
	{
		return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
	}

	__device__ __forceinline__ WORD sig1(WORD x)
	{
		return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
	}

	// GPU SHA-256 functions
	__device__ void sha256_transform_device(SHA256 *ctx, const BYTE *msg);
	__device__ void sha256_device(SHA256 *ctx, const BYTE *msg, size_t len);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif //__SHA256_HEADER__
