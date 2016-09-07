#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cstdio>
#include <string>
#include <sstream>

void printfBoolean(bool *A, int n){
	for(int i = 0; i < n; i++){
		printf("%d%c", A[i] ? 1 : 0, i + 1 == n ? ' ' : '\n');
	}
}

std::string conIntToStr(int x){
	std::string s;
	std::stringstream out;
	out << x;
	return out.str();
}

#endif /* UTILITIES_H_ */
