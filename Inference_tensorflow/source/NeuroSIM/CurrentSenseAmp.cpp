/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cmath>
#include <iostream>
#include <vector>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "CurrentSenseAmp.h"

using namespace std;
extern Param *param;

CurrentSenseAmp::CurrentSenseAmp(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	// TODO Auto-generated constructor stub
	initialized = false;
	invalid = false;
}

CurrentSenseAmp::~CurrentSenseAmp() {
	// TODO Auto-generated destructor stub
}

void CurrentSenseAmp::Initialize(int _numCol, bool _parallel, bool _rowbyrow, double _clkFreq, int _numReadCellPerOperationNeuro) {
	if (initialized)
		cout << "[Current Sense Amp] Warning: Already initialized!" << endl;

	numCol = _numCol;
	parallel = _parallel;
	rowbyrow = _rowbyrow;
	clkFreq = _clkFreq;
	numReadCellPerOperationNeuro = _numReadCellPerOperationNeuro;
	
	widthNmos = MIN_NMOS_SIZE * tech.featureSize;
	widthPmos = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

	double R_start = (double) 1/param->maxConductance;
	double R_index = (double) 1/param->minConductance - (double) 1/param->maxConductance;
	Rref = R_start + (double) R_index/2;
	
	initialized = true;
}

void CurrentSenseAmp::CalculateUnitArea() {
	if (!initialized) {
		cout << "[CurrentSenseAmp] Error: Require initialization first!" << endl;
	} else {
		double hNmosL, wNmosL, hNmosS, wNmosS, hNmosM, wNmosM;
		
		CalculateGateArea(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hNmosS, &wNmosS);
		CalculateGateArea(INV, 1, widthNmos*2, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hNmosM, &wNmosM);
		CalculateGateArea(INV, 1, widthNmos*6, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hNmosL, &wNmosL);
		
		areaUnit = (hNmosL * wNmosL) * 4 + (hNmosS * wNmosS) * 8 + (hNmosM * wNmosM) * 4;
	}
}

void CurrentSenseAmp::CalculateArea(double widthArray) {	// adjust CurrentSenseAmp area by fixing S/A width
	if (!initialized) {
		cout << "[CurrentSenseAmp] Error: Require initialization first!" << endl;
	} else {
		double x = sqrt(areaUnit/HEIGHT_WIDTH_RATIO_LIMIT); // area = HEIGHT_WIDTH_RATIO_LIMIT * x^2
		if (widthArray > x)   // Limit W/H <= HEIGHT_WIDTH_RATIO_LIMIT
			widthArray = x;
		
		area = areaUnit * numCol;
		width = widthArray * numCol;
		height = areaUnit/widthArray;
		
	}
}


void CurrentSenseAmp::CalculateLatency(const vector<double> &columnResistance, double numColMuxed, double numRead) {
	if (!initialized) {
		cout << "[CurrentSenseAmp] Error: Require initialization first!" << endl;
	} else {
		double Group = numCol/numColMuxed;
		double LatencyCol = 0;
		readLatency = 0;
		
		for (double i=0; i<Group; i++) {
			for (double j=0; j<numColMuxed; j++){
				double T_Col = 0;
				T_Col = GetColumnLatency(columnResistance[i*numColMuxed+j]);
				LatencyCol = max(LatencyCol, T_Col);
				if (LatencyCol < 1e-9) {
					LatencyCol = 1e-9;
				} else if (LatencyCol > 20e-9) {
					LatencyCol = 20e-9;
				}
			}
			readLatency += LatencyCol;
		}
		readLatency *= numRead;
		
	}
}

void CurrentSenseAmp::CalculatePower(const vector<double> &columnResistance, double numRead) {
	if (!initialized) {
		cout << "[CurrentSenseAmp] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		for (double i=0; i<numCol; i++) {
			double P_Col = 0, T_Col = 0;
			T_Col = GetColumnLatency(columnResistance[i]);
			P_Col = GetColumnPower(columnResistance[i]);
			readDynamicEnergy += T_Col*P_Col;
		}
		readDynamicEnergy *= numRead;
		
	}
}


double CurrentSenseAmp::GetColumnLatency(double columnRes) {
	double Column_Latency = 0;
	double up_bound = 3, mid_bound = 1.1, low_bound = 0.9;
	double T_max = 0;
	
	if (param->deviceroadmap == 1) {  // HP
		Column_Latency = 1e-9;
	} else {                         // LP
		if (param->technode == 130) {
			T_max = (0.2679*log(columnRes/1000)+0.0478)*1e-9;   // T_max = (0.2679*log(R_BL/1000)+0.0478)*10^-9;

			double ratio = Rref/columnRes;
			double T = 0;
			if (ratio <= low_bound){
				T = T_max * (3.915*pow(ratio,3)-5.3996*pow(ratio,2)+2.4653*ratio+0.3856);  // y = 3.915*x^3-5.3996*x^2+2.4653*x+0.3856;
			} else if (mid_bound <= ratio <= up_bound){
				T = T_max * (0.0004*pow(ratio,4)-0.0087*pow(ratio,3)+0.0742*pow(ratio,2)-0.2725*ratio+1.2211);  // y = 0.0004*x^4-0.0087*x^3+0.0742*x^2-0.2725*x+1.2211;
			} else if (ratio>up_bound){
				T = T_max * (0.0004*pow(ratio,4)-0.0087*pow(ratio,3)+0.0742*pow(ratio,2)-0.2725*ratio+1.2211);
			} else {
				T = T_max;
			}
			Column_Latency = max(Column_Latency, T);
			
		} else if (param->technode == 90) {
			T_max = (0.0586*log(columnRes/1000)+1.41)*1e-9;   // T_max = (0.0586*log(R_BL/1000)+1.41)*10^-9;

			double ratio = Rref/columnRes;
			double T = 0;
			if (ratio <= low_bound){
				T = T_max * (3.726*pow(ratio,3)-5.651*pow(ratio,2)+2.8249*ratio+0.3574);    // y = 3.726*x^3-5.651*x^2+2.8249*x+0.3574;
			} else if (mid_bound <= ratio <= up_bound){
				T = T_max * (0.0000008*pow(ratio,4)-0.00007*pow(ratio,3)+0.0017*pow(ratio,2)-0.0188*ratio+0.9835);  // y = 0.0000008*x^4-0.00007*x^3+0.0017*x^2-0.0188*x+0.9835;
			} else if (ratio>up_bound){
				T = T_max * (0.0000008*pow(ratio,4)-0.00007*pow(ratio,3)+0.0017*pow(ratio,2)-0.0188*ratio+0.9835);
			} else {
				T = T_max;
			}
			Column_Latency = max(Column_Latency, T);
			
		} else if (param->technode == 65) {
			T_max = (0.1239*log(columnRes/1000)+0.6642)*1e-9;   // T_max = (0.1239*log(R_BL/1000)+0.6642)*10^-9;

			double ratio = Rref/columnRes;
			double T = 0;
			if (ratio <= low_bound){
				T = T_max * (1.3899*pow(ratio,3)-2.6913*pow(ratio,2)+2.0483*ratio+0.3202);    // y = 1.3899*x^3-2.6913*x^2+2.0483*x+0.3202;
			} else if (mid_bound <= ratio <= up_bound){
				T = T_max * (0.0036*pow(ratio,4)-0.0363*pow(ratio,3)+0.1043*pow(ratio,2)-0.0346*ratio+1.0512);   // y = 0.0036*x^4-0.0363*x^3+0.1043*x^2-0.0346*x+1.0512;
			} else if (ratio>up_bound){
				T = T_max * (0.0036*pow(ratio,4)-0.0363*pow(ratio,3)+0.1043*pow(ratio,2)-0.0346*ratio+1.0512);
			} else {
				T = T_max;
			}
			Column_Latency = max(Column_Latency, T);
			
		} else if (param->technode == 45) {
			T_max = (0.0714*log(columnRes/1000)+0.7651)*1e-9;    // T_max = (0.0714*log(R_BL/1000)+0.7651)*10^-9;

			double ratio = Rref/columnRes;
			double T = 0;
			if (ratio <= low_bound){
				T = T_max * (3.7949*pow(ratio,3)-5.6685*pow(ratio,2)+2.6492*ratio+0.4807);    // y = 3.7949*x^3-5.6685*x^2+2.6492*x+0.4807
			} else if (mid_bound <= ratio <= up_bound){
				T = T_max * (0.000001*pow(ratio,4)-0.00006*pow(ratio,3)+0.0001*pow(ratio,2)-0.0171*ratio+1.0057);   // 0.000001*x^4-0.00006*x^3+0.0001*x^2-0.0171*x+1.0057;
			} else if (ratio>up_bound){
				T = T_max * (0.000001*pow(ratio,4)-0.00006*pow(ratio,3)+0.0001*pow(ratio,2)-0.0171*ratio+1.0057);
			} else {
				T = T_max;
			}
			Column_Latency = max(Column_Latency, T);
			
		} else {   // technode below and equal to 22nm
			Column_Latency = 1e-9;
		}
	}
	return Column_Latency;
	
}



double CurrentSenseAmp::GetColumnPower(double columnRes) {
	double Column_Power = 0;

	if (param->deviceroadmap == 1) {  // HP
		if (param->technode == 130) {
			Column_Power = (0.00001*(columnRes/1000.0)+8.8898)*1e-6;
		} else if (param->technode == 90) {
			Column_Power = (0.0002*(columnRes/1000.0)+10.09)*1e-6;
		} else if (param->technode == 65) {
			Column_Power = (0.0001*(columnRes/1000.0)+6.9579)*1e-6;
		} else if (param->technode == 45) {
			Column_Power = (0.0037*(columnRes/1000.0)+7.7017)*1e-6;
		} else if (param->technode == 32){  
			Column_Power = (0.0064*(columnRes/1000.0)+7.9648)*1e-6;
		} else {   // technode below and equal to 22nm
			Column_Power = (0.0087*(columnRes/1000.0)+2.1939)*1e-6;
		}
	} else {                         // LP
		if (param->technode == 130) {
			Column_Power = (0.2811*log(columnRes/1000.0)+6.0809)*1e-6;
		} else if (param->technode == 90) {
			Column_Power = (0.0578*log(columnRes/1000.0)+7.6102)*1e-6;
		} else if (param->technode == 65) {
			Column_Power = (0.0710*log(columnRes/1000.0)+6.4147)*1e-6;
		} else if (param->technode == 45) {
			Column_Power = (0.0710*log(columnRes/1000.0)+6.4147)*1e-6;
		} else if (param->technode == 32){  
			Column_Power = (0.0251*log(columnRes/1000.0)+4.7835)*1e-6;
		}  else {   // technode below and equal to 22nm
			Column_Power = (0.0516*log(columnRes/1000.0)+3.2349)*1e-6;
		}
	}
	
	return Column_Power;
	
}


void CurrentSenseAmp::PrintProperty(const char* str) {
	//cout << "Current Sense Amplifier Properties:" << endl;
	FunctionUnit::PrintProperty(str);
}


