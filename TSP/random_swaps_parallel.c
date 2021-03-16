/* 
Compiling: gcc random_swaps_parallel.c -o random_swaps_parallel -O2 -fopenmp
Executing: time ./random_swaps_parallel
Output:
Starting Distance = 3348790584
Final Distance = 91334084

Time of execution (on a 8-core linux system):

real    0m57,027s
user    7m24,745s
sys     0m3,603s
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>


#define N 1000			// Size of the grid
#define NODES 10000		// Number of nodes
#define SWAPS 10000000	// Number of swaps

// City struct using short ints
typedef struct city{
	short int x;
	short int y;
} CITY; 

// Global variables
CITY Route[NODES];				// Depicts the current route
unsigned int currentDistance;

// Initialize the Route with cities
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
		Route[i].x = x;
		Route[i].y = y;
	}
}

// Find the square of the euclidean distance between two nodes
unsigned int nodeDistance(CITY c1, CITY c2){
	unsigned int dist;
	int distx = c1.x - c2.x;
	int disty = c1.y - c2.y;
	dist = distx * distx + disty * disty;
	return dist;
}

// Find the route distance (using the square of the euclidean) 
unsigned int routeDistance(){
	unsigned int dist = 0;
	#pragma omp parallel for reduction(+:dist)
	for(int i = 0; i < NODES-1; i++)
		dist += nodeDistance(Route[i], Route[i+1]);
	dist += nodeDistance(Route[NODES-1], Route[0]);
	return dist;
}

// Execute a random swap
void randomSwap(){
	int index1, index2;
	unsigned int newDistance;
	CITY temp;
	
	// Choose two cities to swap.
	// Indexes are in range (1, NODES-1)
	index1 = (rand() % (NODES-1)) + 1;
	do{
		index2 = (rand() % (NODES-1)) + 1;
	} while(index1 == index2);
	
	// Swap
	temp = Route[index1];
	Route[index1] = Route[index2];
	Route[index2] = temp;
	
	// Check for improvement
	newDistance = routeDistance();
	if(newDistance < currentDistance){
		currentDistance = newDistance;
		return;
	}
	
	// Swap back if the route became worse
	temp = Route[index1];
	Route[index1] = Route[index2];
	Route[index2] = temp;
}

int main(int argc, char *argv[]) {
	createCities();
	currentDistance = routeDistance();
	printf("Starting Distance = %u\n", currentDistance);
	for(int i = 0; i < SWAPS; i++)
		randomSwap();
	printf("Final Distance = %u\n", currentDistance);
	return 0;
}
