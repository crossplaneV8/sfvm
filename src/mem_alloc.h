
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>


// memory pool allocator
struct sf_allocator;


// create a new allocator
struct sf_allocator *sf_create_allocator(void);


// discard an existing allocator
void sf_discard_allocator(struct sf_allocator *alloc);


// allocate memory from allocator
void *sf_malloc(struct sf_allocator *alloc, size_t size);


// return memory back to allocator
void sf_free(void *buf);


// increase ref cnt of shared memory
void sf_shared_memory_inc(void *buf);


// decrease ref cnt of shared memory, free memory when ref cnt < 1
void sf_shared_memory_dec(void *buf);


#ifdef __cplusplus
    }
#endif

