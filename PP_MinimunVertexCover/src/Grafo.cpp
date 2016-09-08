#ifndef GRAFO_CPP
#define GRAFO_CPP

#include <sstream>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "Grafo.h"
#include "Utilities.h"

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

void Graph::addEdgesUnd(int orig, int dest){
    node *nodeDest = newNode(dest);
    vert[orig].origen = orig;
    nodeDest->next = vert[orig].rootUnDi;
    vert[orig].rootUnDi = nodeDest;
    vert[orig].gradoUnDi++;
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

bool Graph::levantarGrafo(const char* nameFile, bool isCSV, int iniUno) {
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
    int limit = 122, k = 0;
    for(;getline(myFile, line); ){
    	int orig, dest;
    	if(isCSV){
			int pos = line.find(';');
			orig =  atoi(line.substr(0, 3).c_str()) - iniUno;
			dest = atoi(line.substr(pos + 1, line.size()).c_str()) - iniUno;
    	}else{
    		istringstream iss(line);
    		iss >> orig;
    		iss >> dest;
    		orig -= iniUno;
    		dest -= iniUno;
    	}
//    	printf("%d %d\n", orig, dest);
        if(orig != dest){
			addEdges(orig, dest);
			addEdges(dest, orig);

			addEdgesUnd(orig, dest);
        }
    }
    printf("termino de cargar \n", nameFile);
    return true;
}

void Graph::genFileForVisualization(const char* nameFile, bool *mvc){

	string cabe1 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
	string graphmlIni = "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n";
	string colorNode = "<key id=\"d0\" for=\"node\" attr.name=\"color\" attr.type=\"string\"> <default>yellow</default> </key>\n";
	string graphIni = "<graph id=\"G\" edgedefault=\"undirected\">\n";
	string graphFin = "</graph>\n";
	string graphmlFin = "</graphml>\n";
	string keyColor = "<data key=\"d0\">green</data>\n";
	string nodeSimple[] = {"<node id=\"n", "\"/>\n"};
	string nodeColored[] = {"<node id=\"n", "\">\n", "</node>\n"};
	string edge[] = {"<edge id=\"e", "\" source=\"n", "\" target=\"n", "\"/>\n"};

	ofstream file;
	file.open(nameFile);
	file << cabe1;
	file << graphmlIni;
	file << colorNode;
	file << graphIni;
	for(int tid = 0; tid < numVert; tid++){
		string strTid = conIntToStr(tid);
		if(mvc[tid]){
			file << nodeColored[0] + strTid + nodeColored[1];
			file << keyColor;
			file << nodeColored[2];
		}else{
			file << nodeSimple[0] + strTid + nodeSimple[1];
		}
	}

	for(int tid = 0, eid = 0; tid < numVert; tid++){
		if(vert[tid].gradoUnDi){
			string strVSrc = conIntToStr(tid);
			node* temp = vert[tid].rootUnDi;
			for(int j = 0; j < vert[tid].gradoUnDi; j++){
				string strEid = conIntToStr(eid++);
				string strVDes = conIntToStr(temp->valor);
				file << edge[0] + strEid + edge[1] + strVSrc + edge[2] + strVDes + edge[3];
				temp = temp->next;
			}
		}
	}

	file << graphFin;
	file << graphmlFin;

	file.close();
}

#endif /* GRAFO_CPP */
