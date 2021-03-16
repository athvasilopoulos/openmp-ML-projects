/* 
Compiling: gcc HH_serial.c -o HH_serial -O2
Executing: time ./HH_serial
Output:
Minimum Distance = 2756252

Time of execution:

real    0m34,860s
user    0m34,849s
sys     0m0,000s
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
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
// and choose who to go to next, based on alpha
int nearestNeighbour(int index){
	static float p2 = 1.0 / (alpha + 1.0);
	unsigned int mindist1 = UINT_MAX;
	int minpos1, minpos2, mindist2;
	for(int i = 1; i < NODES; i++){
		if (Cities[i].available){
			unsigned int dist = nodeDistance(Cities[Route[index]], Cities[i]);
			if(dist < mindist1){
				minpos2 = minpos1;
				mindist2 = mindist1;
				minpos1 = i;
				mindist1 = dist;
			}
			else if(dist < mindist2){
				minpos2 = i;
				mindist2 = dist;
			}
		}
	}
	
	// Choose the second nearest with probability p2
	if(index < NODES-2){
		float p = rand() / (float)RAND_MAX;
		if (p < p2){
			currentDistance += mindist2;
			Cities[minpos2].available = false;
			return minpos2;
		}
	}
	
	// Choose the nearest with probability 1-p2
	currentDistance += mindist1;
	Cities[minpos1].available = false;
	return minpos1;
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
	printf("Minimum Distance = %d\n", minDistance);
	return 0;
}
