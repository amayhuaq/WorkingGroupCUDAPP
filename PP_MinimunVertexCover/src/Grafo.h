#ifndef GRAFO_H_
#define GRAFO_H_

struct node{
  int valor;
  node* next;
};

struct listNode{
	int grado;
	int gradoUnDi;
	int origen;
	int *veci;
	int *vecSinRep;
	node* root;
	node* rootUnDi;
	int posIniNei;
	listNode(){ grado = 0; origen = 0; root = NULL; gradoUnDi = 0; rootUnDi = NULL;}
};

class Graph{
	public:

		listNode* vert;
		int *listNeight, numVert, numEdges;
		Graph();
		node* newNode(int dest);
		void printGrafo();
		void printGrafoRefinado();
		void addEdges(int orig, int dest);
		void addEdgesUnd(int orig, int dest);
		void refinarGrafo();
		void compactarGrafo();
		bool levantarGrafo(const char* nameFile, bool isCSV, int iniUno);
		void genFileForVisualization(const char* nameFile, bool *mvc);
};

#endif /* GRAFO_H_ */
