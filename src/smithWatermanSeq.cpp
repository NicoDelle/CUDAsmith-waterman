#include "smithWatermanSeq.h"
#include <fstream>


int __max4(int n1, int n2, int n3, int n4)
{
	int tmp1, tmp2;
	tmp1 =  n1 > n2 ? n1 : n2;
	tmp2 = n3 > n4 ? n3 : n4;
	tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
	return tmp1;
}

void backtrace(u_int16_t *simple_rev_cigar, u_int16_t **dir_mat, int i, int j, int max_cigar_len)
{
	int n;
	for (n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
	{
		int dir = dir_mat[i][j];
		if (dir == 1 || dir == 2)
		{
			i--;
			j--;
		}
		else if (dir == 3)
			i--;
		else if (dir == 4)
			j--;

		simple_rev_cigar[n] = dir;
	}
}

u_int16_t **smithWatermanSeq(char **query, char **reference, u_int16_t **cigar)
{
    int ins = -2, del = -2, match = 1, mismatch = -1; // penalties
    u_int32_t **sc_mat = (u_int32_t **) malloc((S_LEN + 1) * sizeof(u_int32_t *));
    for (int i = 0; i < (S_LEN + 1); i++)
		sc_mat[i] = (u_int32_t *)malloc((S_LEN + 1) * sizeof(u_int32_t));
    
    u_int16_t **dir_mat = (u_int16_t **)malloc((S_LEN + 1) * sizeof(u_int16_t *));
	for (int i = 0; i < (S_LEN + 1); i++)
		dir_mat[i] = (u_int16_t *)malloc((S_LEN + 1) * sizeof(u_int16_t *));

    int *res = (int *)malloc(N * sizeof(int));
    int p;
    for (int n = 0; n < N; n++)
    {
        int max = ins; // in sw all scores of the alignment are >= 0, so this will be for sure changed
        int maxi, maxj;
        // initialize the scoring matrix and direction matrix to 0
        for (int i = 0; i < S_LEN + 1; i++)
        {
            for (int j = 0; j < S_LEN + 1; j++)
            {
                sc_mat[i][j] = 0;
                dir_mat[i][j] = 0;
            }
        }
        
        // compute the alignment
        for (int i = 1; i < S_LEN + 1; i++)
        {
            for (int j = 1; j < S_LEN + 1; j++)
            {
                // compare the sequences characters
                //-> PER OGNI RIGA, otteniamo se elementi di query e reference corrispondono
                int comparison = (query[n][i - 1] == reference[n][j - 1]) ? match : mismatch;
                // compute the cell knowing the comparison result
                int tmp = __max4(
                    sc_mat[i - 1][j - 1] + comparison, 
                    sc_mat[i - 1][j] + del, 
                    sc_mat[i][j - 1] + ins, 
                    0
                );
                char dir;

                //scegliamo la direzione da intraprendere una volta applicate delle penalità/premi.
                if (tmp == (sc_mat[i - 1][j - 1] + comparison))
                    dir = ((comparison == match )? 1 : 2); //-> se max è match dir=1, altrimenti se mismatch dir=2
                else if (tmp == (sc_mat[i - 1][j] + del))
                    dir = 3; 						  //-> se max è del, allora dir=3
                else if (tmp == (sc_mat[i][j - 1] + ins))
                    dir = 4;					  	 //-> se max è ins, allora dir=4
                else
                    dir = 0;						//-> se il risultato della cella sarebbe minore di 0, dir=0

                dir_mat[i][j] = dir;
                sc_mat[i][j] = tmp; //in sc si mette il massimo, in dir il valore calcolato

                if (tmp > max) //si tiene traccia della cella di valore massimo
                {
                    max = tmp;
                    maxi = i;
                    maxj = j;
                }
                
            }
        }
        res[n] = sc_mat[maxi][maxj];//il miglior match finisce sempre nella cella con lo score più alto
        backtrace(cigar[n], dir_mat, maxi, maxj, S_LEN * 2);

        if (n == 0)
        {
            std::cout << "#" << n << ": " << std::endl;
            std::cout << "Max coords: " << maxi << ", " << maxj << std::endl;
            std::cout << "Max val: " << max << std::endl << std::endl;

            // Store the last computed sc_mat to a file
            std::ofstream outFile("scoring_matrix.txt");
            if (outFile.is_open())
            {
                for (int i = 0; i < S_LEN + 1; i++)
                {
                    for (int j = 0; j < S_LEN + 1; j++)
                    {
                        outFile << sc_mat[i][j] << (j == S_LEN ? "\n" : ",");
                    }
                }
                outFile.close();
            }
            else
            {
                std::cerr << "Unable to open file for writing" << std::endl;
            }
        }
    }

    return dir_mat;
}