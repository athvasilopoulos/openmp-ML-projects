/* 
Compiling: gcc HH_parallel.c -o HH_parallel -O2 -fopenmp
Executing: time ./HH_parallel
Output:
Minimum Distance = 2756252

Time of execution (on a 8-core linux system):

real    0m8,353s
user    1m5,155s
sys     0m0,438s
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <omp.h>
#include <string.h>

#define N 1000		// Size of the grid
#define NODES 10000	// Number of nodes
#define REP 100		// Number of repetitions
#define alpha 6		// Probability variable

// City struct using short ints
typedef struct city{
	short int x;
	short int y;
	bool available;
} CITY; 

// Global variables
CITY Cities[NODES];		// Cities created		
short int Route[NODES];	// Indexes to Cities showing the route
unsigned int currentDistance = 0;

// Initialize the Cities array
void createCities(){
	bool coords[N][N];
	memset(coords, true, N*N*sizeof(bool));
	short int x, y;
	for (int i = 0; i < NODES; i++){
		do{
			x = rand() % N;
			y = rand() % N;
		} while(!coords[x][y]);
		coords[x][y] = false;
		Cities[i].x = x;
		Cities[i].y = y;
		Cities[i].available = true;
	}
	Cities[0].available = false;
}

// Find the square of the euclidean distance between two nodes
unsigned int nodeDistance(CITY c1, CITY c2){
	unsigned int dist;
	int distx = c1.x - c2.x;
	int disty = c1.y - c2.y;
	dist = distx * distx + disty * disty;
	return dist;
}

// Find the two nearest neighbours of a node 
// in a parallel way and choose who to go to next
short int nearestNeighbour(int index){
	static float p2 = 1.0 / (alpha + 1.0);
	// The arrays where each thread will store its 2 minimums
	unsigned int mindist[16];
	short int minpos[16];
	// Initialize the mindist
	memset(mindist, UINT_MAX, 16 * sizeof(unsigned int));
	#pragma omp parallel num_threads(8)
	{
		int id;
		int i, step, start, stop, num_threads;
		unsigned int my_min[2];
		short int my_minpos[2];
		
		// Create the start/stop indexes
		num_threads = omp_get_num_threads();
		step = NODES / num_threads;
		id = omp_get_thread_num();
		start = id * step;
		if(id != (num_threads - 1))
			stop = start + step;
		else
			stop = NODES;
			
		my_min[0] = mindist[2*id];
		my_min[1] = mindist[2*id+1];
		
		// Find the 2 minimums in the specified range
		for(i = start; i < stop; i++)
			if(Cities[i].available){
				unsigned int dist = nodeDistance(Cities[Route[index]], Cities[i]);
				if(dist < my_min[0]){
					my_min[1] = my_min[0];
					my_minpos[1] = my_minpos[0];
					my_min[0] = dist;
					my_minpos[0] = i;
				}
				else if(dist < my_min[1]){
					my_min[1] = dist;
					my_minpos[1] = i;
				}
			}
		// Place the minimums in the specified arrays 
		mindist[2*id] = my_min[0];
		mindist[2*id+1] = my_min[1];
		minpos[2*id] = my_minpos[0];
		minpos[2*id+1] = my_minpos[1];
	}
	
	unsigned int global_mindist[2] = {mindist[0], mindist[1]};
	short int global_minpos[2] = {minpos[0], minpos[1]};
	// Find the 2 global minimums using the fact that the
	// local minimums are already sorted
	for(int i = 1; i < 8; i++){
		if(mindist[2*i] < global_mindist[0]){
			if(mindist[2*i+1] < global_mindist[0]){
				global_mindist[1] = mindist[2*i+1];
				global_minpos[1] = minpos[2*i+1];
				global_mindist[0] = mindist[2*i];
				global_minpos[0] = minpos[2*i];
			}
			else{
				global_mindist[1] = global_mindist[0];
				global_minpos[1] = global_minpos[0];
				global_mindist[0] = mindist[2*i];
				global_minpos[0] = minpos[2*i];
			}	
		}
		else if(mindist[2*i] < global_mindist[1]){
			global_mindist[1] = mindist[2*i];
			global_minpos[1] = minpos[2*i];
		}
	}
	
	// Choose the second nearest with probability p2 
	if(index < NODES-2){
		float p = rand() / (float)RAND_MAX;
		if(p < p2){
			currentDistance += global_mindist[1];
			Cities[global_minpos[1]].available = false;
			return global_minpos[1];
		}
	}
	
	// Choose the nearest with probability 1-p2
	currentDistance += global_mindist[0];
	Cities[global_minpos[0]].available = false;
	return global_minpos[0];
}

int main(int argc, char *argv[]) {
	createCities();
	Route[0] = 0;
	unsigned int minDistance = UINT_MAX;
	// Execute the algorithm REP times and hold the minimum distance
	for (int j = 0; j < REP; j++){
		currentDistance = 0;
		for (int i = 1; i < NODES; i++)
			Cities[i].available = true;
		for (int i = 0; i < NODES-1; i++)
			Route[i+1] = nearestNeighbour(i);
		currentDistance += nodeDistance(Cities[Route[NODES-1]], Cities[Route[0]]);
		if(currentDistance < minDistance)
			minDistance = currentDistance;
	}
	printf("Minimum Distance = %u\n", minDistance);
	return 0;
}
