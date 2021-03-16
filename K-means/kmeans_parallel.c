/* 
Compiling: gcc kmeans_parallel.c -o kmeans_parallel -O3 -fopenmp
Executing: time ./kmeans_parallel
Time of execution:

real    0m9,595s
user    0m58,986s
sys     0m0,188s
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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
    int i;
    float totaldist = 0.0f;
	#pragma omp parallel for reduction(+:totaldist)	
    for (i = 0; i < N; i++)
    {
        float mindist = euclDist(Vec[i], Center[0]);
        int minpos = 0;
        for (int j = 1; j < Nc; j++)
        {
            float dist = euclDist(Vec[i], Center[j]);
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
    int i, counters[Nc];
    memset(Center, 0, sizeof(Center));
    memset(counters, 0, sizeof(counters));
    for (int i = 0; i < N; i++)
    {
        int j = Classes[i];
        counters[j]++;
        #pragma omp simd
        for (int k = 0; k < Nv; k++)
        {
            Center[j][k] += Vec[i][k];
        }
    }
    for (i = 0; i < Nc; i++)
    {
        float f = 1.0 / counters[i];
        #pragma omp simd
        for (int j = 0; j < Nv; j++)
            Center[i][j] *= f;
    }
}

// *************************************
// Calculates the square of the euclidean distance
// between two vectors.
float euclDist(float *a, float *b)
{
    int i;
    float dist = 0.0;
    #pragma omp simd reduction(+:dist)
    for (i = 0; i < Nv; i++)
    {
        float t = a[i] - b[i];
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
