#ifndef GRAFO_H_
#define GRAFO_H_

struct node{
  int valor;
  node* next;
};

struct listNode{
	int grado;
	int origen;
	int *veci;
	node* root;
	int posIniNei;
	listNode(){ grado = 0; origen = 0; root = NULL;}
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
		void refinarGrafo();
		void compactarGrafo();
		bool levantarGrafo(const char* nameFile);
};

#endif /* GRAFO_H_ */
