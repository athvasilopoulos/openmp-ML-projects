/* 
Compiling: gcc kmeans_serial.c -o kmeans_serial -O2
Executing: time ./kmeans_serial
Time of execution:

real    3m40,092s
user    3m39,638s
sys     0m0,269s
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// *************************************
#define N 100000 // Number of samples
#define Nv 1000  // Number of dimensions
#define Nc 100   // Number of centers

// *************************************
void createData(void);
void createCenters(void);
float classification(void);
void estimateCenters(void);
float euclDist(float *a, float *b);

// *************************************
float Vec[N][Nv];     // Data array
float Center[Nc][Nv]; // Centers array
int Classes[N];       // Classification array

// *************************************
// Creates random data.
// The range is (-2, 2) for every dimension
void createData(void)
{
    int i, j;
    for (i = 0; i < N; i++)
        for (j = 0; j < Nv; j++)
            Vec[i][j] = 4 * (rand() / (float)RAND_MAX - 0.5);
}

// *************************************
// Initializes the centers by choosing random
// samples from the data
void createCenters(void)
{
    int i, j, nc, P[Nc];
    P[0] = rand() % N;
    for (i = 1; i < Nc; i++)
    {
        do
        {
            nc = rand() % N;
            for (j = 0; j < i; j++)
            {
                if (nc == P[j])
                    break;
            }
        } while (j < i);
        P[i] = nc;
    }
    for (i = 0; i < Nc; i++)
        memcpy(Center[i], Vec[P[i]], sizeof(float) * Nv);
}

// *************************************
// Classifies each sample on the nearest center and
// then returns the total distance of all the samples
// from their centers.
float classification(void)
{
    int i, j, minpos;
    float mindist, dist, totaldist = 0.0f;
    for (i = 0; i < N; i++)
    {
        mindist = euclDist(Vec[i], Center[0]);
        minpos = 0;
        for (j = 1; j < Nc; j++)
        {
            dist = euclDist(Vec[i], Center[j]);
            if (dist < mindist)
            {
                mindist = dist;
                minpos = j;
            }
        }
        totaldist += mindist;
        Classes[i] = minpos;
    }
    return totaldist;
}

// *************************************
// Calculates the new centers for the next step
void estimateCenters(void)
{
    int i, j, k, counters[Nc];
    memset(Center, 0, sizeof(Center));
    memset(counters, 0, sizeof(counters));
    for (j = 0; j < N; j++)
    {
        int i = Classes[j];
        counters[i]++;
        for (k = 0; k < Nv; k++)
        {
            Center[i][k] += Vec[j][k];
        }
    }
    for (i = 0; i < Nc; i++)
    {
        float f = 1.0 / counters[i];
        for (j = 0; j < Nv; j++)
            Center[i][j] *= f;
    }
}

// *************************************
// Calculates the square of the euclidean distance
// between two vectors.
float euclDist(float *a, float *b)
{
    int i;
    float dist = 0.0, t;
    for (i = 0; i < Nv; i++)
    {
        t = a[i] - b[i];
        dist += t * t;
    }
    return dist;
}

// *************************************
int main(void)
{
    int c = 0;
    float dist, prevdist, dif;
    // Initialize data
    createData();
    createCenters();
    // First classification
    dist = classification();
    // 15 more K-means steps
    do
    {
        estimateCenters();
        prevdist = dist;
        dist = classification();
        dif = prevdist - dist;
        c++;
        printf("%f\n", dif);
    } while (c < 16);
    return 0;
}