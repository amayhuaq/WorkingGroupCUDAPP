#ifndef MVC_SERIAL_CPP
#define MVC_SERIAL_CPP

#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include "MVC_Serial.h"

MVCSerial::MVCSerial(Graph gVal){
	g = gVal;
	nNodes = g.numVert;
	nNodesMVC = 0;
	timeExe = 0.0;
	mvc = NULL;
	prevMvc = NULL;
	adj = NULL;
	arrayNodMVC = NULL;
	terminedSerial = false;
}

void MVCSerial::kerner1MVCSerial(){
	int mDegSerial, deg2Serial;
	for(int tid = 0; tid < nNodes; tid++){
		mDegSerial = g.vert[tid].grado;
		for(int i = 0;i < g.vert[tid].grado; i++){
			deg2Serial = g.vert[g.vert[tid].veci[i]].grado;
			mDegSerial = mDegSerial < deg2Serial ? mDegSerial : deg2Serial;
		}
		if(mDegSerial == g.vert[tid].grado)
			mvc[tid] = false;
	}
}

void MVCSerial::kernel2MVCSerial(){
	for(int tid = 0; tid < nNodes; tid++){
		adj[tid] = true;
		for(int i = 0; i < g.vert[tid].grado; i++){
			if(!mvc[g.vert[tid].veci[i]])
				adj[tid] = false;
		}
		if(mvc[tid] != !adj[tid]){
			terminedSerial = false;
		}

		prevMvc[tid] = mvc[tid];
	}
}

void MVCSerial::kernel3MVCSerial(){
	for(int tid = 0; tid < nNodes; tid++){
		for(int i = 0, eid; i < g.vert[tid].grado; i++){
			eid = g.vert[tid].veci[i];
			if(prevMvc[eid] && !adj[eid] && adj[tid])
				mvc[tid] = false;
		}
	}
}

void MVCSerial::kernel4MVCSerial(){
	for(int tid = 0; tid < nNodes; tid++){
		if(!prevMvc[tid] && !adj[tid]){
			for(int i = 0, eid; i < g.vert[tid].grado; i++){
				eid = g.vert[tid].veci[i];
				if(!prevMvc[eid] && !adj[eid] && eid < tid)
					mvc[tid] = true;
			}
		}
	}
}

void MVCSerial::ejecutarSerial(){
	timeExe = 0.0;
	mvc = new bool[nNodes];
	prevMvc = new bool[nNodes];
	adj = new bool[nNodes];

	for(int i = 0; i < nNodes; i++){
		mvc[i] = true;
		adj[i] = true;
		prevMvc[i] = true;
	}

	clock_t time = clock();
	kerner1MVCSerial();
	kernel2MVCSerial();
	while(!terminedSerial){
		terminedSerial = true;
		kernel3MVCSerial();
		kernel2MVCSerial();
		kernel4MVCSerial();
		kernel2MVCSerial();
	}
	time = clock() - time;
	timeExe = ((float)time) / CLOCKS_PER_SEC;

	for(int i = 0; i < nNodes; i++)
		nNodesMVC += mvc[i];
}

float MVCSerial::getTimeExe(){
	return timeExe;
}

int MVCSerial::getnNodesMVC(){
	return nNodesMVC;
}

int* MVCSerial::getListNodesMVC(){
	arrayNodMVC = new int[nNodesMVC];
	for(int i = 0, j = 0; i < nNodes; i++){
		if(mvc[i])
			arrayNodMVC[j++] = i;
	}
	return arrayNodMVC;
}

#endif
