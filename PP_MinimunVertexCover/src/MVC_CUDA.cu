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
int nEjecCUD, nEjecZero;

struct structFile{
	string nameFile;
	bool isCSV;
	int iniUno;
	structFile(string a, bool b, int c){ nameFile = a; isCSV = b; iniUno = c;}
};


__device__ int getIdVertex() {
	return threadIdx.x + blockIdx.x * blockDim.x;
}

//__global__ void kernel1_mvc(int nNodes, listNode *nodes, int *listNeigh, bool *mvc) {
__global__ void kernel1_mvc(int nNodes, nodeSimple *nodes, int *listNeigh, bool *mvc) {
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

//__global__ void kernel2_mvc(int nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc, bool *terminated) {
__global__ void kernel2_mvc(int nNodes, nodeSimple *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc, bool *terminated) {
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

//__global__ void kernel3_mvc(int nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc) {
__global__ void kernel3_mvc(int nNodes, nodeSimple *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc) {
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

//__global__ void kernel4_mvc(int nNodes, listNode *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc) {
__global__ void kernel4_mvc(int nNodes, nodeSimple *nodes, int *listNeigh, bool *mvc, bool *adj, bool *prevMvc) {
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
	//listNode *nodes;
	nodeSimple *nodes;
	int *listNeigh;
	int nNodes = grafo->numVert;
	int nEdges = grafo->numEdges;

	nEjecZero = 0;

	// variables para device
	//listNode *devNodes;
	nodeSimple *devNodes;
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
	//cudaHostAlloc((void**)&nodes, nNodes * sizeof(listNode), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&nodes, nNodes * sizeof(nodeSimple), cudaHostAllocWriteCombined | cudaHostAllocMapped);
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
		//nodes[i] = grafo->vert[i];
		nodes[i] = grafo->listNodeSimple[i];
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
		nEjecZero++;
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

	cudaFreeHost(mvc);
	cudaFreeHost(adj);
	cudaFreeHost(prevMvc);
	cudaFreeHost(listNeigh);
	cudaFreeHost(nodes);
	cudaFreeHost(&terminated);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime / 1000.0;
}

float ejecutarCUDA(Graph* grafo) {
	// variables para host
	bool *adj, terminated = false;
	int nNodes = grafo->numVert;
	int nEdges = grafo->numEdges;

	nEjecCUD = 0;

	// variables para device
	//listNode *devNodes;
	nodeSimple *devNodes;
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
	//cudaMalloc((void**)&devNodes, nNodes * sizeof(listNode));
	cudaMalloc((void**)&devNodes, nNodes * sizeof(nodeSimple));
	cudaMalloc((void**)&devListNeig, nEdges * sizeof(int));

	cudaMemcpy(devTerminated, &terminated, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devMvc, mvc, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devPrevMvc, mvc, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(devAdj, adj, nNodes * sizeof(bool), cudaMemcpyHostToDevice);
	//cudaMemcpy(devNodes, grafo->vert, nNodes * sizeof(listNode), cudaMemcpyHostToDevice);
	cudaMemcpy(devNodes, grafo->listNodeSimple, nNodes * sizeof(nodeSimple), cudaMemcpyHostToDevice);
	cudaMemcpy(devListNeig, grafo->listNeight, nEdges * sizeof(int), cudaMemcpyHostToDevice);

	int blocks = (nNodes + MAX_THREADS_BY_BLOCK - 1) / MAX_THREADS_BY_BLOCK;
	int threads = MAX_THREADS_BY_BLOCK;

	kernel1_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc);
	kernel2_mvc<<<blocks, threads>>>(nNodes, devNodes, devListNeig, devMvc, devAdj, devPrevMvc, devTerminated);
	while(!terminated){
		nEjecCUD++;
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

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime / 1000.0;
}

int main() {
	string path = "data/";
	string resPath = "res/";
	vector<structFile> arrayFiles;
//	arrayFiles.push_back(structFile("randomGraph4.csv", 1, 1));
//	arrayFiles.push_back(structFile("randomGraph6_01.csv", 1, 1));
//	arrayFiles.push_back(structFile("randomGraph7_01.csv", 1, 1));
//	arrayFiles.push_back(structFile("randomGraph7_02.csv", 1, 1));
//	arrayFiles.push_back(structFile("randomGraph10.csv", 1, 1));
	arrayFiles.push_back(structFile("randomGraph10000.csv", 1, 1));
	arrayFiles.push_back(structFile("p2p-Gnutella31.txt", 0, 0));
	arrayFiles.push_back(structFile("networkGraph_20000_600000.csv", 1, 1));
//	arrayFiles.push_back(structFile("web-BerkStan_685230_7600595.txt", 0, 1));
//	arrayFiles.push_back(structFile("soc-LiveJournal1.txt", 0, 0));

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
		g->levantarGrafo((path + arrayFiles[i].nameFile).c_str(), arrayFiles[i].isCSV, arrayFiles[i].iniUno);
		g->refinarGrafo();
		g->compactarGrafo();

		// Ejecutando version CUDA con global memory
		elapsedTime1 = ejecutarCUDA(g);
		g->genFileForVisualization((resPath + arrayFiles[i].nameFile + ".graphml").c_str(), mvc);

		// Ejecutando version CUDA con zero-memory
		elapsedTime2 = ejecutarCUDAZeroCopy(g);

		// Ejecutando version SERIAL
		MVCSerial mvcSerial(*g);
		mvcSerial.ejecutarSerial();
		bool *arrayMVCSerial = mvcSerial.getListNodesMVC();
		int nNodesMVCSerial = mvcSerial.getnNodesMVC();

		printf("Graph: %s - nVertex: %d\n", arrayFiles[i].nameFile.c_str(), g->numVert);
		printf("> Times: CUDA GloMem = %f secs, CUDA ZeroMem = %f secs, Serial = %f secs. \n", elapsedTime1, elapsedTime2, mvcSerial.getTimeExe());
		printf("%d %d %d\n", nEjecCUD, nEjecZero, mvcSerial.nEjec);
	}
	return 0;
}
