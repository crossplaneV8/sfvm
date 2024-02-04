
#include "base_struct.h"
#include "mem_alloc.h"

#define NUM_RANK    ((int)32)
#define MIN_SIZE    ((size_t)64)


// head of a memory block
struct mem_block
{
    struct sf_list *recycler;   // where to return this memory block
    int64_t ref_cnt;            // reference counter for shared memory
};


// memory pool allocator
struct sf_allocator
{
    struct sf_list *all, *idle[NUM_RANK];
};


// (MIN_SIZE << rank) >= size
static int _get_rank(size_t size)
{
    size_t space = MIN_SIZE;
    int rank = 0;

    while (space < size) {
        space *= 2;
        rank++;
    }
    return rank;
}


// create a new allocator
struct sf_allocator *sf_create_allocator(void)
{
    struct sf_allocator *alloc = malloc(sizeof(struct sf_allocator));
    alloc->all = sf_create_list();

    for (int i=0; i<NUM_RANK; i++) {
        alloc->idle[i] = sf_create_list();
    }
    return alloc;
}


// discard an existing allocator
void sf_discard_allocator(struct sf_allocator *alloc)
{
    for (int i=0; i<alloc->all->cnt; i++) {
        free(alloc->all->buf[i]);
    }
    for (int i=0; i<NUM_RANK; i++) {
        sf_discard_list(alloc->idle[i]);
    }
    sf_discard_list(alloc->all);
    free(alloc);
}


// allocate memory from allocator
void *sf_malloc(struct sf_allocator *alloc, size_t size)
{
    if (size > 0) {
        const int rank = _get_rank(size);
        struct sf_list *list = alloc->idle[rank];
        struct mem_block *block;

        if (list->cnt > 0) {
            block = list->buf[--(list->cnt)];
        } else {
            block = malloc(sizeof(struct mem_block) + (MIN_SIZE << rank));
            sf_list_append(alloc->all, block);
        }
        block->recycler = list;
        block->ref_cnt = 0;
        return block + 1;
    }
    return NULL;
}


// return memory back to allocator
void sf_free(void *buf)
{
    if (buf != NULL) {
        struct mem_block *block = (struct mem_block*)buf - 1;
        sf_list_append(block->recycler, block);
    }
}


// increase ref cnt of shared memory
void sf_shared_memory_attach(void *buf)
{
    if (buf != NULL) {
        struct mem_block *block = (struct mem_block*)buf - 1;
        block->ref_cnt += 1;
    }
}


// decrease ref cnt of shared memory, free memory when ref cnt < 1
void sf_shared_memory_detach(void *buf)
{
    if (buf != NULL) {
        struct mem_block *block = (struct mem_block*)buf - 1;
        block->ref_cnt -= 1;
        if (block->ref_cnt < 1) {
            sf_list_append(block->recycler, block);
        }
    }
}


