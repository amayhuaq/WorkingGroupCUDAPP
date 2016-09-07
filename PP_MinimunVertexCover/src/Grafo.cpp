#ifndef GRAFO_CPP
#define GRAFO_CPP

#include <sstream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include "Grafo.h"

using namespace std;

Graph::Graph(void){
	vert = NULL;
	listNeight = NULL;
	numVert = 0;
	numEdges = 0;
}

node* Graph::newNode(int dest){
    node* resNode = new node;
    resNode->valor = dest;
    resNode->next = NULL;
    return resNode;
}

void Graph::printGrafo(){
    printf("numVertices: %d\n", numVert);
    for(int i = 0; i < numVert; i++){
        printf("vert(%d) grado(%d): ", vert[i].origen, vert[i].grado);
        node* temp = vert[i].root;
        for(int j = 0; temp != NULL; j++){
            printf("%d%c", temp->valor, j + 1 == vert[i].grado ? '\n' : ' ');
            temp = temp->next;
        }
    }
}

void Graph::printGrafoRefinado(){
    printf("numVertices: %d\n", numVert);
    for(int i = 0; i < numVert; i++){
        printf("vert(%d) grado(%d): ", vert[i].origen, vert[i].grado);
        for(int j = 0; j < vert[i].grado; j++){
            printf("%d%c", vert[i].veci[j], j + 1 == vert[i].grado ? '\n' : ' ');
        }
    }

    for(int i = 0; i < numVert; i++)
        printf("%d%c", vert[i].posIniNei, i + 1 == numVert ? '\n': ' ');

    for(int i = 0; i < numEdges; i++)
        printf("%d%c", listNeight[i], i + 1 == numEdges ? '\n' : ' ');
}

void Graph::addEdges(int orig, int dest){
    node *nodeDest = newNode(dest);
    vert[orig].origen = orig;
    nodeDest->next = vert[orig].root;
    vert[orig].root = nodeDest;
    vert[orig].grado++;
    numEdges++;
}

void Graph::refinarGrafo(){
    for(int i = 0; i < numVert; i++){
        vert[i].veci = new int[vert[i].grado];
        node* temp = vert[i].root;
        for(int j = 0; j < vert[i].grado; j++){
           vert[i].veci[j] = temp->valor;
           temp = temp->next;
        }
    }
}

void Graph::compactarGrafo() {
    listNeight = new int[numEdges];
    for(int i = 0, pos = 0; i < numVert; i++){
        vert[i].posIniNei = pos;
        for(int j = 0; j < vert[i].grado; j++){
            listNeight[pos++] = vert[i].veci[j];
        }
    }
}

bool Graph::levantarGrafo(const char* nameFile) {
    ifstream myFile(nameFile);
    if(!myFile.is_open()) {
    	printf("No se encontro el archivo indicado");
    	return false;
    }

    string line;
    getline(myFile, line);
    istringstream iss(line);
    iss >> numVert;

    vert = new listNode[numVert];
    numEdges = 0;
    for(int i = 0; i < numVert; i++)
        vert[i] = listNode();

    for(;getline(myFile, line); ){
        int pos = line.find(';');
        int orig =  atoi(line.substr(0, 3).c_str()) - 1;
        int dest = atoi(line.substr(pos + 1, line.size()).c_str()) - 1;
        addEdges(orig, dest);
        addEdges(dest, orig);
    }
    return true;
}

#endif /* GRAFO_CPP */
