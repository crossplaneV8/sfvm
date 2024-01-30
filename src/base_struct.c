
#include "base_struct.h"


// create an empty list
struct sf_list *sf_create_list(void)
{
    struct sf_list *list = malloc(sizeof(struct sf_list));
    list->len = 32;
    list->cnt = 0;
    list->buf = malloc(list->len * sizeof(void*));
    return list;
}


// discard an existing list
void sf_discard_list(struct sf_list *list)
{
    if (list != NULL) {
        if (list->buf != NULL) {
            free(list->buf);
        }
        free(list);
    }
}


// append an object to the list
void sf_list_append(struct sf_list *list, void *obj)
{
    if (list->cnt == list->len) {
        const int len = 2 * list->len;
        void **buf = malloc(len * sizeof(void*));
        memcpy(buf, list->buf, list->len * sizeof(void*));
        free(list->buf);
        list->buf = buf;
        list->len = len;
    }
    list->buf[list->cnt++] = obj;
}


// get list index of a given object
int sf_list_find(struct sf_list *list, void *obj)
{
    for (int i=0; i<list->cnt; i++) {
        if (list->buf[i] == obj) {
            return i;
        }
    }
    return -1;
}


// key-value pair
struct _hash_pair
{
    void *key;
    void *val;
};


// row of pairs
struct _hash_row
{
    int len, cnt;
    struct _hash_pair buf[];
};


// hash table
struct sf_dict
{
    int hash_bits, item_cnt;
    struct _hash_row **rows;
};


// hash function for pointers
static int _ptr_hash(int hash_bits, void *ptr)
{
    uint64_t k = 11400714819323198549ull;
    uint64_t n = k * (uint64_t)ptr;
    return (int)(n >> (64 - hash_bits));
}


// create an empty hash table
struct sf_dict *sf_create_dict(void)
{
    const int hash_bits = 5;
    const int hash_size = 1 << hash_bits;
    struct sf_dict *dict = malloc(sizeof(struct sf_dict));
    dict->hash_bits = hash_bits;
    dict->item_cnt = 0;
    dict->rows = malloc(hash_size * sizeof(void*));
    memset(dict->rows, 0, hash_size * sizeof(void*));
    return dict;
}


// discard an existing hash table
void sf_discard_dict(struct sf_dict *dict)
{
    int hash_size = 1 << dict->hash_bits;
    for (int i=0; i<hash_size; i++) {
        if (dict->rows[i] != NULL) {
            free(dict->rows[i]);
        }
    }
    free(dict->rows);
    free(dict);
}


// clear hash table
void sf_clear_dict(struct sf_dict *dict)
{
    int hash_size = 1 << dict->hash_bits;
    for (int i=0; i<hash_size; i++) {
        if (dict->rows[i] != NULL) {
            dict->rows[i]->cnt = 0;
        }
    }
    dict->item_cnt = 0;
}


static struct _hash_row *_create_hash_row(int len)
{
    struct _hash_row *row = malloc(sizeof(struct _hash_row) +
                                   len * sizeof(struct _hash_pair));
    row->len = len;
    row->cnt = 0;
    return row;
}


static struct _hash_row *_extend_hash_row(struct _hash_row *old)
{
    struct _hash_row *row = _create_hash_row(2 * old->len);
    row->cnt = old->cnt;
    memcpy(row->buf, old->buf, old->cnt * sizeof(struct _hash_pair));
    return row;
}


static void _extend_dict(struct sf_dict *dict)
{
    int old_size = 1 << dict->hash_bits++;
    int new_size = 1 << dict->hash_bits;
    struct _hash_row **old_rows = dict->rows;
    dict->rows = malloc(new_size * sizeof(void*));
    memset(dict->rows, 0, new_size * sizeof(void*));
    dict->item_cnt = 0;

    for (int i=0; i<old_size; i++) {
        struct _hash_row *row = old_rows[i];
        if (row != NULL) {
            for (int j=0; j<row->cnt; j++) {
                sf_write_dict(dict, row->buf[j].key, row->buf[j].val);
            }
            free(row);
        }
    }
    free(old_rows);
}


// write hash table
void sf_write_dict(struct sf_dict *dict, void *key, void *val)
{
    int h = _ptr_hash(dict->hash_bits, key);
    struct _hash_row *row = dict->rows[h];

    if (row != NULL) {
        for (int i=0; i<row->cnt; i++) {
            if (row->buf[i].key == key) {
                row->buf[i].val = val; return;
            }
        }
        if (row->cnt == row->len) {
            dict->rows[h] = _extend_hash_row(row);
            free(row);
        }
    } else {
        dict->rows[h] = _create_hash_row(8);
    }
    row = dict->rows[h];
    row->buf[row->cnt].key = key;
    row->buf[row->cnt].val = val;
    row->cnt++;

    if (++(dict->item_cnt) > (1 << dict->hash_bits) * 64) {
        _extend_dict(dict);
    }
}


// read hash table
void *sf_read_dict(struct sf_dict *dict, void *key)
{
    int h = _ptr_hash(dict->hash_bits, key);
    struct _hash_row *row = dict->rows[h];

    if (row != NULL) {
        for (int i=0; i<row->cnt; i++) {
            if (row->buf[i].key == key) {
                return row->buf[i].val;
            }
        }
    }
    return NULL;
}




