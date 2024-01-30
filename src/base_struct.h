
#pragma once


#ifdef __cplusplus
    extern "C" {
#endif


#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


// list of objects
struct sf_list
{
    int len, cnt;
    void **buf;
};


// create an empty list
struct sf_list *sf_create_list(void);

// discard an existing list
void sf_discard_list(struct sf_list *list);

// append an object to the list
void sf_list_append(struct sf_list *list, void *obj);

// get list index of a given object
int sf_list_find(struct sf_list *list, void *obj);


// hash table
struct sf_dict;

// create an empty hash table
struct sf_dict *sf_create_dict(void);

// discard an existing hash table
void sf_discard_dict(struct sf_dict *dict);

// clear hash table
void sf_clear_dict(struct sf_dict *dict);

// write hash table
void sf_write_dict(struct sf_dict *dict, void *key, void *val);

// read hash table
void *sf_read_dict(struct sf_dict *dict, void *key);



#ifdef __cplusplus
    }
#endif

