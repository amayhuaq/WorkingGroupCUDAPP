/*
 * MVC_Serial.h
 *
 *  Created on: 06/09/2016
 *      Author: cesar
 */

#ifndef MVC_SERIAL_H_
#define MVC_SERIAL_H_
#include "Grafo.h"

class MVCSerial{
	private:
		Graph g;
		bool *adj, *prevMvc, *mvc;
		bool terminedSerial;
		int nNodes, nNodesMVC, *arrayNodMVC;
		float timeExe;
	public:
		MVCSerial(Graph gVal);
		void ejecutarSerial();
		void kerner1MVCSerial();
		void kernel2MVCSerial();
		void kernel3MVCSerial();
		void kernel4MVCSerial();
		float getTimeExe();
		int getnNodesMVC();
		bool* getListNodesMVC();
};

#endif /* MVC_SERIAL_H_ */
