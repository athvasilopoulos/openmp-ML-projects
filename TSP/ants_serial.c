/* 
Compiling: gcc ants_serial.c -o ants_serial -O2 -lm
Executing: time ./ants_serial
Output:
Distance = 97029.424021

Time of execution:

real    20m37,048s
user    20m35,808s
sys     0m0,456s
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

// Problem parameters
#define N 1000			// Size of the grid
#define NODES 10000		// Number of nodes
#define REPS 20			// Number of "days"
#define NUM_OF_ANTS 16	// Number of ants
#define alpha 1			// Parameter a
#define beta 5			// Parameter b
#define evaporation 0.5	// Evaporation parameter
#define Q 100			// Parameter Q

// City struct using short ints
typedef struct city{
	short int x;
	short int y;
} CITY;

// Ant struct with the memory arrays
typedef struct ant{
	int trail[NODES];
	bool unvisited[NODES];
} ANT;

// Helping functions declarations
double probabilityNorm(int currentCity, int ant);	
double moveProbability(int i,int j,double norm);
int binSearch(double *arr, int high, double num);
float nodeDistance(CITY c1, CITY c2);
double trailDistance(int ant);

// Global arrays
CITY Cities[NODES];				// Cities created
float Distances[NODES][NODES];	// Distance between every city
float T[NODES][NODES];			// Pheromone array
ANT ants[NUM_OF_ANTS];			// Ants array

// Initialize the global arrays
void createData(){
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
	}
	
	// Calculate distance between every other node
	// and put the starting pheromone level an every edge
	for(int i = 0; i < NODES; i++){
		for(int j = 0; j <= i; j++){
			Distances[i][j] = nodeDistance(Cities[i], Cities[j]);
			T[i][j] = 1.0;
		}
	}
	
	// Copy the upper triangle of 
	// the arrays to the lower one
	for(int i = 0; i < NODES; i++){
		for(int j = i + 1; j < NODES; j++){
			Distances[i][j] = Distances[j][i];
			T[i][j] = 1.0;
		}
	}
	
	// Initialize every ant, on a random city
	for(int i = 0; i < NUM_OF_ANTS; i++){
		memset(ants[i].unvisited, true, NODES*sizeof(bool));
		int starting_city = rand() % NODES;
		ants[i].trail[0] = starting_city;
		ants[i].unvisited[starting_city] = false;
	}
}

// Reposition the ant in a random node for the next day
void antReset(){
	for(int i = 0; i < NUM_OF_ANTS; i++){
		memset(ants[i].unvisited, true, NODES*sizeof(bool));
		int starting_city = rand() % NODES;
		ants[i].trail[0] = starting_city;
		ants[i].unvisited[starting_city] = false;
	}
}

// Decide which edge an ant follows next, using roulette wheel selection
void antStep(int ant, int trailSize){
	int currentCity = ants[ant].trail[trailSize-1];
	int unvisitedLength = NODES - trailSize;
	double norm = probabilityNorm(currentCity, ant);
	double cumulativeSum = 0.0;
	double cumulativeProb[unvisitedLength];
	int nextCity[unvisitedLength];
	int pointer = 0;
	for(int i = 0; i < NODES; i++){
		if(ants[ant].unvisited[i]){
			double p = moveProbability(currentCity,i,norm);
			cumulativeSum += p;
			cumulativeProb[pointer] = cumulativeSum;
			nextCity[pointer] = i;
			pointer++;
		}
	}
	float gp = rand()/(float)RAND_MAX;
	int position = binSearch(cumulativeProb, unvisitedLength, gp);
	ants[ant].trail[trailSize] = nextCity[position];
	ants[ant].unvisited[nextCity[position]] = false;
}

// Find the shortest of the ants' trails and
// also deposit the new pheromone for each ant
double measureTrails(){
	double minTour;
	for (int i = 0; i < NUM_OF_ANTS; i++){
		double tour = trailDistance(i);
		if(i == 0 || tour < minTour)
			minTour = tour;

		double depositAmount = Q / tour;
		for (int j = 0; j < NODES-1; j++)
			T[ants[i].trail[j]][ants[i].trail[j+1]] += depositAmount;
		T[ants[i].trail[NODES-1]][ants[i].trail[0]] += depositAmount;
	}
	return minTour;
}

// Evaporate a percentage of the pheromone
void evaporate(){
	float coeff = 1 - evaporation;
	for(int i = 0; i < NODES; i++)
		for(int j = 0; j < NODES; j++)
			T[i][j] *= coeff;
}

int main(int argc, char *argv[]) {
	createData();
	double minTour;
	for (int n = 0; n < REPS; n++){
		// Ants complete their routes
		for (int i = 0; i < NUM_OF_ANTS; i++)
			for (int j = 1; j < NODES; j++)
				antStep(i, j);
		// Evaporate old pheromone
		evaporate();
		// Find minimum tour for this batch and put pheromone
		double repMinTour = measureTrails();
		if(n == 0 || repMinTour < minTour)
			minTour = repMinTour;
		// Reposition ants for the next day
		antReset();
		printf("Minimum in %d rep is: %lf\n", n, minTour);
	}
	printf("Distance = %lf\n", minTour);
	return 0;
}

// Find the euclidean distance between two nodes
float nodeDistance(CITY c1, CITY c2){
	float dist;
	float distx = c1.x - c2.x;
	float disty = c1.y - c2.y;
	dist = sqrtf(distx * distx + disty * disty);
	return dist;
}

// Calculate the distance of an ant's trail
double trailDistance(int ant){
	double dist = 0.0;
	for(int i = 0; i < NODES-1; i++)
		dist += Distances[ants[ant].trail[i]][ants[ant].trail[i+1]];
	dist += Distances[ants[ant].trail[NODES-1]][ants[ant].trail[0]];
}

// Calculate the normalize factor for a
// specific ant using the available edges
double probabilityNorm(int currentCity, int ant) {
	double norm = 0.0;
	for(int i = 0; i < NODES; i++)
		if(ants[ant].unvisited[i])
			norm += (pow(T[currentCity][i],alpha)) * (pow((1.0 / Distances[currentCity][i]),beta));
	return norm;
}

// Calculate the probability the ant
// chooses the edge connecting cities i and j
double moveProbability(int i,int j,double norm) {
	double p = (pow(T[i][j],alpha))*(pow((1.0 / Distances[i][j]),beta));
	p /= norm;
	return p;
}

// Return the position that num would
// be inserted into in a sorted array arr
int binSearch(double *arr, int high, double num){
	int l = 0;
	int h = high;
	int mid;
	while(l < h){
		mid = (l+h) / 2;
		if(arr[mid] > num) h = mid;
		else l = mid + 1;
	}
	return l;
}
