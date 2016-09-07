#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cstdio>


void printfBoolean(bool *A, int n){
	for(int i = 0; i < n; i++){
		printf("%d%c", A[i] ? 1 : 0, i + 1 == n ? ' ' : '\n');
	}
}


#endif /* UTILITIES_H_ */
