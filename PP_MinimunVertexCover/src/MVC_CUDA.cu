#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <string>
#include <vector>
#include "Grafo.h"
#include "MVC_Serial.h"

using namespace std;

#define MAX_THREADS_BY_BLOCK 1024

bool *mvc;

__device__ int getIdVertex() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

__global__ void kernel1_mvc(int nNodes, listNode *nodes, int *listNeigh, bool *mvc) {
	int tid = getIdVertex();
	if(tid < nNodes){
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

__global__ void kernel2_mvc(int nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc, bool *terminated) {
	int tid = getIdVertex();
	if(tid < nNodes){
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

__global__ void kernel3_mvc(int nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc) {
	int tid = getIdVertex(), eid;
	if(tid < nNodes){
		int nEdges = nodes[tid].grado;
		int posVec = nodes[tid].posIniNei;

		for(int i = 0; i < nEdges; i++) {
			eid = listNeigh[posVec + i];
			if(prevMvc[eid] && !adj[eid] && adj[tid])
				mvc[tid] = false;
		}
	}
}

__global__ void kernel4_mvc(int nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc) {
	int tid = getIdVertex();
	if(tid < nNodes){
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

float ejecutarCUDAZeroCopy(Graph *grafo) {
	// variables para host
	bool *adj, *prevMvc, *terminated;
	listNode *nodes;
	int *listNeigh;
	int nNodes = grafo->numVert;
	int nEdges = grafo->numEdges;

	// variables para device
	listNode *devNodes;
	bool *devMvc, *devPrevMvc, *devAdj, *devTerminated;
	int *devListNeig;

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaHostAlloc((void**)&terminated, sizeof(bool), cudaHostAllocMapped);
	cudaHostAlloc((void**)&mvc, nNodes * sizeof(bool), cudaHostAllocMapped);
	cudaHostAlloc((void**)&adj, nNodes * sizeof(bool), cudaHostAllocMapped);
	cudaHostAlloc((void**)&prevMvc, nNodes * sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&nodes, nNodes * sizeof(listNode), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&listNeigh, nEdges * sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped);

	cudaHostGetDevicePointer(&devTerminated, terminated, 0);
	cudaHostGetDevicePointer(&devMvc, mvc, 0);
	cudaHostGetDevicePointer(&devAdj, adj, 0);
	cudaHostGetDevicePointer(&devPrevMvc, prevMvc, 0);
	cudaHostGetDevicePointer(&devNodes, nodes, 0);
	cudaHostGetDevicePointer(&devListNeig, listNeigh, 0);

	// Se asignan los valores iniciales de cada variable
	*terminated = false;
	for(uint i = 0; i < nNodes; i++) {
		nodes[i] = grafo->vert[i];
		mvc[i] = true;
		adj[i] = true;
		prevMvc[i] = true;
	}
	for(uint i = 0; i < nEdges; i++) {
		listNeigh[i] = grafo->listNeight[i];
	}

	int blocks = (nNodes + MAX_THREADS_BY_BLOCK - 1) / MAX_THREADS_BY_BLOCK;
	int threads = MAX_THREADS_BY_BLOCK;

	kernel1_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc);
	cudaThreadSynchronize();
	kernel2_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
	cudaThreadSynchronize();
	while(!(*terminated)) {
		*terminated = true;
		kernel3_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc);
		cudaThreadSynchronize();
		kernel2_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
		cudaThreadSynchronize();
		kernel4_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc);
		cudaThreadSynchronize();
		kernel2_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
		cudaThreadSynchronize();
	}

//	cout << "CUDA Zero Memory\n";
//	for(uint i = 0; i < nNodes; i++) {
//		cout << "Node " << i << " - Mvc " << mvc[i] << " Adj " << adj[i] << endl;
//	}

	cudaFreeHost(mvc);
	cudaFreeHost(adj);
	cudaFreeHost(prevMvc);
	cudaFreeHost(listNeigh);
	cudaFreeHost(nodes);
	cudaFreeHost(&terminated);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	return elapsedTime/1000.0;
}

float ejecutarCUDA(Graph* grafo) {
	// variables para host
	bool *adj, terminated = false;
	int nNodes = grafo->numVert;
	int nEdges = grafo->numEdges;

	// variables para device
	listNode *devNodes;
	bool *devMvc, *devPrevMvc, *devAdj, *devTerminated;
	int *devListNeig;

	mvc = new bool[nNodes];
	adj = new bool[nNodes];

	for(uint i = 0; i < nNodes; i++) {
		mvc[i] = true;
		adj[i] = true;
	}

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&devTerminated, sizeof(bool));
	cudaMalloc((void**)&devMvc, nNodes * sizeof(bool));
	cudaMalloc((void**)&devPrevMvc, nNodes * sizeof(bool));
	cudaMalloc((void**)&devAdj, nNodes * sizeof(bool));
	cudaMalloc((void**)&devNodes, nNodes * sizeof(listNode));
	cudaMalloc((void**)&devListNeig, nEdges * sizeof(int));

	cudaMemcpy(devTerminated, &terminated, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devMvc, mvc, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devPrevMvc, mvc, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devAdj, adj, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devNodes, grafo->vert, nNodes * sizeof(listNode), cudaMemcpyHostToDevice);
	cudaMemcpy(devListNeig, grafo->listNeight, nEdges * sizeof(int), cudaMemcpyHostToDevice);

	int blocks = (nNodes + MAX_THREADS_BY_BLOCK - 1) / MAX_THREADS_BY_BLOCK;
	int threads = MAX_THREADS_BY_BLOCK;

	kernel1_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc);
	kernel2_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
	while(!terminated){
		terminated = true;
		cudaMemcpy(devTerminated, &terminated, sizeof(bool), cudaMemcpyHostToDevice);
		kernel3_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc);
		kernel2_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
		kernel4_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc);
		kernel2_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
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

//	cout << "CUDA Global Memory\n";
//	for(uint i = 0; i < nNodes; i++) {
//		cout << "Node " << i << " - Mvc " << mvc[i] << " Adj " << adj[i] << endl;
//	}

	return elapsedTime / 1000.0;
}

int main() {
	string path = "data/";
	string resPath = "res/";
	vector<string> arrayFiles;
	arrayFiles.push_back("randomGraph4.csv");
	arrayFiles.push_back("randomGraph7_01.csv");
	arrayFiles.push_back("randomGraph7_02.csv");
	arrayFiles.push_back("randomGraph10.csv");
	arrayFiles.push_back("randomGraph10000.csv");

	int whichDevice;
	cudaDeviceProp prop;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);
	if(prop.canMapHostMemory != 1) {
		cout << "Device no puede mapear memoria en CPU" << endl;
		return 0;
	}
	cudaSetDeviceFlags(cudaDeviceMapHost);

	float elapsedTime1, elapsedTime2;
	for(int i = 0; i < arrayFiles.size(); i++) {
		Graph* g = new Graph();
		g->levantarGrafo((path + arrayFiles[i]).c_str());
		g->refinarGrafo();
		g->compactarGrafo();

		// Ejecutando version CUDA con global memory
		elapsedTime1 = ejecutarCUDA(g);
		g->genFileForVisualization((resPath + arrayFiles[i] + ".graphml").c_str(), mvc);

		// Ejecutando version CUDA con zero-memory
		elapsedTime2 = ejecutarCUDAZeroCopy(g);

		// Ejecutando version SERIAL
		MVCSerial mvcSerial(*g);
		mvcSerial.ejecutarSerial();
		bool *arrayMVCSerial = mvcSerial.getListNodesMVC();
		int nNodesMVCSerial = mvcSerial.getnNodesMVC();

		printf("Graph: %s - nVertex: %d\n", arrayFiles[i].c_str(), g->numVert);
		printf("> Times: CUDA GloMem = %f secs, CUDA ZeroMem = %f secs, Serial = %f secs.\n", elapsedTime1, elapsedTime2, mvcSerial.getTimeExe());
	}
	return 0;
}
