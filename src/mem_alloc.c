
#include "base_struct.h"
#include "mem_alloc.h"

#define NUM_RANK    ((int)32)
#define MIN_SIZE    ((size_t)64)


// head of a memory block
struct mem_block
{
    // size = MIN_SIZE << rank
    int rank, _pad[3];
};


// memory pool allocator
struct sf_allocator
{
    struct sf_list *all, *idle[NUM_RANK];
};


// get rank by size
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


// allocate a new memory block from system
static struct mem_block *_create_mem_block(int rank)
{
    struct mem_block *block = malloc(sizeof(struct mem_block) + (MIN_SIZE << rank));
    block->rank = rank;
    return block;
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

        if (list->cnt > 0) {
            struct mem_block *block = list->buf[--(list->cnt)];
            return block + 1;
        } else {
            struct mem_block *block = _create_mem_block(rank);
            sf_list_append(alloc->all, block);
            return block + 1;
        }
    }
    return NULL;
}


// return memory back to allocator
void sf_free(struct sf_allocator *alloc, void *buf)
{
    if ((alloc != NULL) && (buf != NULL)) {
        struct mem_block *block = (struct mem_block*)buf - 1;
        sf_list_append(alloc->idle[block->rank], block);
    }
}

