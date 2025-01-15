#include <iostream>

#define N 1000
#define S_LEN 512

int __max4(int a, int b, int c, int d);
void backtrace(u_int16_t *simple_rev_cigar, u_int16_t **dir_mat, int i, int j, int max_cigar_len);
void smithWatermanSeq(char **query, char **reference, u_int16_t **cigar);