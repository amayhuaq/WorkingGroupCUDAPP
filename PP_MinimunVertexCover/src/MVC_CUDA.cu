#include <stdio.h>
#include <iostream>
#include "Grafo.h"
#include "MVC_Serial.h"

using namespace std;

#define MAX_THREADS_BY_BLOCK 1024

int nNodesMCVCUDA, *arrayMvcCUDA;
float elapsedTime;

__global__ void knowData(long *x1, long *y1, long *z1, long *x2, long *y2, long *z2){
	*x1 = (long)blockDim.x;
	*y1 = (long)blockDim.y;
	*z1 = (long)blockDim.z;

	*x2 = (long)gridDim.x;
	*y2 = (long)gridDim.y;
	*z2 = (long)gridDim.z;

}

__device__ int getIdVertex() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void kernel1_mvc(int* nNodes, listNode *nodes, int *listNeigh, bool *mvc) {
	int tid = getIdVertex();
	if(tid < *nNodes){
		int deg = nodes[tid].grado;
		int posVec = nodes[tid].posIniNei;
		int tempDeg, mdeg = deg;

		for(int i = 0; i < deg; i++){
			tempDeg = nodes[listNeigh[posVec + i]].grado;
			mdeg = min(mdeg, tempDeg);
		}
		if(deg == mdeg)
			mvc[tid] = false;
	}
}

__global__ void kernel2_mvc(int* nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc, bool *terminated) {
	int tid = getIdVertex();
	if(tid < *nNodes){
		int nEdges = nodes[tid].grado;
		int posVec = nodes[tid].posIniNei;

		adj[tid] = true;
		for(int i = 0; i < nEdges; i++)
			if(!mvc[listNeigh[posVec + i]])
				adj[tid] = false;
		if(mvc[tid] != !adj[tid])
			*terminated = false;
		prevMvc[tid] = mvc[tid];
	}
}

__global__ void kernel3_mvc(int *nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc) {
	int tid = getIdVertex(), eid;
	if(tid < *nNodes){
		int nEdges = nodes[tid].grado;
		int posVec = nodes[tid].posIniNei;

		for(int i = 0; i < nEdges; i++) {
			eid = listNeigh[posVec + i];
			if(prevMvc[eid] && !adj[eid] && adj[tid])
				mvc[tid] = false;
		}
	}
}

__global__ void kernel4_mvc(int *nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc) {
	int tid = getIdVertex();
	if(tid < *nNodes){
		if(!prevMvc[tid] && !adj[tid]){
			int nEdges = nodes[tid].grado, eid;
			int posVec = nodes[tid].posIniNei;

			for(int i = 0; i < nEdges; i++){
				eid = listNeigh[posVec + i];
				if(!prevMvc[eid] && !adj[eid] && eid < tid)
					mvc[tid] = true;
			}
		}
	}
}

void ejecutarCUDA(Graph* grafo) {
	// variables host
	bool *adj, *mvc, terminated = false;
	int nNodes = grafo->numVert;
	int nEdges = grafo->numEdges;

	// variables devices
	listNode *devNodes;
	bool *devMvc, *devPrevMvc, *devAdj, *devTerminated;
	int *devListNeig, *devNumNodes;

	mvc = new bool[nNodes];
	adj = new bool[nNodes];

	for(uint i = 0; i < nNodes; i++) {
		mvc[i] = true;
		adj[i] = true;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&devNumNodes, sizeof(int));
	cudaMalloc((void**)&devTerminated, sizeof(bool));
	cudaMalloc((void**)&devMvc, nNodes * sizeof(bool));
	cudaMalloc((void**)&devPrevMvc, nNodes * sizeof(bool));
	cudaMalloc((void**)&devAdj, nNodes * sizeof(bool));
	cudaMalloc((void**)&devNodes, nNodes * sizeof(listNode));
	cudaMalloc((void**)&devListNeig, nEdges * sizeof(int));

	cudaMemcpy(devNumNodes, &nNodes, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devTerminated, &terminated, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devMvc, mvc, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devPrevMvc, mvc, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devAdj, adj, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devNodes, grafo->vert, nNodes * sizeof(listNode), cudaMemcpyHostToDevice);
	cudaMemcpy(devListNeig, grafo->listNeight, nEdges * sizeof(int), cudaMemcpyHostToDevice);

	int blocks = (nNodes + MAX_THREADS_BY_BLOCK - 1) / MAX_THREADS_BY_BLOCK;
	int threads = MAX_THREADS_BY_BLOCK;

	kernel1_mvc<<<blocks, threads>>>(devNumNodes, devNodes, devListNeig, devMvc);
	kernel2_mvc<<<blocks, threads>>>(devNumNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
	while(!terminated){
		terminated = true;
		cudaMemcpy(devTerminated, &terminated, sizeof(bool), cudaMemcpyHostToDevice);
		kernel3_mvc<<<blocks, threads>>>(devNumNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc);
		kernel2_mvc<<<blocks, threads>>>(devNumNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
		kernel4_mvc<<<blocks, threads>>>(devNumNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc);
		kernel2_mvc<<<blocks, threads>>>(devNumNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
		cudaMemcpy(&terminated, devTerminated, sizeof(bool), cudaMemcpyDeviceToHost);
	}

	cudaMemcpy(mvc, devMvc, nNodes * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(adj, devAdj, nNodes * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaFree(devMvc);
	cudaFree(devPrevMvc);
	cudaFree(devAdj);
	cudaFree(devListNeig);
	cudaFree(devNodes);
	cudaFree(devTerminated);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	for(uint i = 0; i < nNodes; i++)
	{
		cout << "Node " << i << " - Mvc " << mvc[i] << " Adj " << adj[i] << endl;
	}

	for(int i = 0; i < nNodes; i++)
		nNodesMCVCUDA += mvc[i];
	arrayMvcCUDA = new int[nNodesMCVCUDA];
	for(int i = 0, j = 0; i < nNodes; i++)
		if(mvc[i])
			arrayMvcCUDA[j++] = i;
}

int main() {
	Graph* g = new Graph();
	if(g->levantarGrafo("data/randomGraph7_02.csv"))
	{
		g->refinarGrafo();
		g->compactarGrafo();

		ejecutarCUDA(g);
		printf("ElapsedtimeCUDA  : %f\n", elapsedTime / 1000.0);

	}
	return 0;
}
