#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h> 
#include <chrono>
//#include <curses.h>
#include <unistd.h>
#include <iomanip>

using namespace std;

// some common function not specific for neural nets
// sigmoid activation function
double activation(double x)
{
    return 1.0/(1.0 + exp(-x));
}

// returns derivative of sigmoid activation function
double d_activation(double x){
    return (1.0 - activation(x))*activation(x);
 }  
 

// takes array a of size n and sets
// its elements to random values 
// in min-max range
void Randomize(double* a, int n, double min, double max){
	//cout<<"RAND_MAX="<<RAND_MAX<<endl;
    srand (time(NULL));
    for ( int i = 0 ; i < n ; i++){
        double f = (double)rand() / RAND_MAX;
        a[i]= min + f * (max - min);
    }
}

// prints n elements of array a on the screen
void PrintVector(double* a, int n){
    for ( int i = 0 ; i < n ; i++){
        cout<<a[i]<<" ";
    }
    cout<<endl;
}
// class contains variables and functions
// of the neural net
class NN{
	private:
	  int nInputs; // number of inputs
	  int nOutputs; //number of outputs
	  int nHiddenNeurons; // number of neurons in hidden layer
	 
	  // search parameters
	  double dw;   // bias and weights step for gradient estimation
	  double learningRate; // hmm, it is learning  rate
	  int nStepsMax;
	  
	  // whole training set 
	  double* inputsTraining;   // inputs - column, pattern - row 
	  double* answersTraining;  // outputs - column, pattern - row
	  int nTrainingEntries;  // number of rows in training set
	  
	  // working set
	  double* inputsWorking;   // inputs - column, pattern - row 
	  int nWorkingEntries;  // number of rows in training set
	  
	  // current NN situation
	  double* currentInputs;  // row picked from inputsTraining
	  double* currentAnswers; // ditto for training answers 
	  double* currentOutputs; // guess what? 
	  double* currentError;   // current differnce between answers and output
	  //output error for current dataset row
	  double sumOfOutputErrors;
	  // sum of errors for all dataset entries
	  double totalDatasetError;
	  
	  
	  // input to hidden layer
	  double* weightsHidden;  // hidden layer weights matrix Wh
	  double* biasHidden;     // hidden layer bias vector bh
	  double* d_weightsHidden;  // hidden layer weights matrix Wh derivative
	  double* d_biasHidden;     // hidden layer bias vector bh derivative
	  double* deltaHidden;  // for backpropagation
	  
	  // state of the hidden layer
	  double* netHidden;     // yh = Wh*z + bh - renaming 
	  double* outHidden;   //  zh = activ(Wh*x + bh) - layer output
	  
      // hidden layer to output layer
      double* weightsOutput;  // Wh
      double* biasOutput;     // bh
      double* d_weightsOutput;  // dWh/dw - gradients
      double* d_biasOutput;     // dbh / dw - gradients
      double* deltaOutput;     // dbh / dw - gradients
 
	  // state of the output layer
      double* nnSumOutput;  
      double* netOutput;  // yo = Wo*zh + bo
      double* nnBiaOutput;
	  double* nnOutOutput; // zo = activ(yo)
	  
	public:
	  NN(){ }; // constructor
	  int LoadTrainingSet(string file,int nInp,int nHid , int nOut);
	  int LoadWorkingSet(string file,int nInp);
	  void DisplayTrainDigit(int nImage); // ASCII art display
	  void DisplayWorkDigit(int nImage); // ASCII art display
	  int InitNet(double min, double max);
	  void GetTrainingEntry(int iTrainRow);
	  void GetWorkingEntry(int iWorkRow);
	  void ForwardProp();
	  double GetTotalError(){ return sumOfOutputErrors;};
	  void PrintErr(){ cout<<" Error: "<<sumOfOutputErrors<<endl;};
	  void DirectGradientEstimation();
	  void BackProp();
	  void StepByGradient();
	  double GetOutputError(){ return sumOfOutputErrors;};
	  double TotalDatasetError(); // sum of errors for all rows of train data
	  void Train1();
	  void PrintOutputs();
      void DisplayResults();
      void stepBackProp();	
};

// loads inputsTraining and answersTraining from "file"
int NN::LoadTrainingSet(string file,int nInp,int nHid , int nOut){
	std::ifstream data(file);
    std::string line;
    nTrainingEntries = 0;
    nInputs = nInp;
    nOutputs = nOut;
    nHiddenNeurons  = nHid;
    // count number of lines in input file
    while(std::getline(data,line)) { nTrainingEntries++; }
    cout<<" Thera are "<<nTrainingEntries<<" entries in dataset"<<endl;
    // reserve the memory
    inputsTraining = new double[nTrainingEntries*nInputs];
    answersTraining = new double[nTrainingEntries*nOutputs];
    cout<<" Memory reserved..."<<endl;
    // rewind the file
    data.clear();
    data.seekg(0);
    // read training data file
    for(int iim = 0; iim<nTrainingEntries; iim++) {
		std::getline(data,line);
    	//cout<<" iim= "<<iim<<" Input: "<<line<<endl;
        std::stringstream lineStream(line);
        std::string cell;
        int count = 0;
        // break input string into inputs and answers
        while(std::getline(lineStream,cell,' ')) {
            //cout<<"count="<<count<<"cell="<<cell<<" "<<endl;
            if (count<nInputs) { 
				inputsTraining[iim*nInputs+count] = atof(cell.c_str()) ;
				//cout<<" count="<<count<<" Inp[][]="<<inputsTraining[iim*nInputs+count]<<endl;
			} else {
				answersTraining[iim*nOutputs+count-nInputs] = atof(cell.c_str());
				//cout<<" count-nInputs="<<count-nInputs<<" Out[][]="<<answersTraining.GetElement(iim,count-nInputs)<<endl;
			}	
          count++;
        } // while
        //cout<<" Input string "<<iim<<" parsed"<<endl;
        //char stop;
        //cin>>stop; 
    } // for
    cout<<" Training set loaded. Inputs:"<<endl;
    //char stop;
    //cin>>stop; 
    //inputsTraining.PrintMatrix();
    data.close();
	return 0;
}

// loads working set inputs from "file"
int NN::LoadWorkingSet(string file,int nInp){
	std::ifstream data(file);
    std::string line;
    nWorkingEntries = 0;
    nInputs = nInp;
    // count number of lines in input file
    while(std::getline(data,line)) { nWorkingEntries++; }
    cout<<" Thera are "<<nWorkingEntries<<" entries in dataset (working)"<<endl;
    // reserve the memory
    inputsWorking = new double[nTrainingEntries*nInputs];
    cout<<" Memory reserved..."<<(int)nTrainingEntries*nInputs<<endl;
    // rewind the file
    data.clear();
    data.seekg(0);
    // read working data file
    for(int iim = 0; iim<nTrainingEntries; iim++) {
		std::getline(data,line);
        std::stringstream lineStream(line);
        std::string cell;
        int count = 0;
        // break input string into inputs and answers
        while(std::getline(lineStream,cell,' ')) {
            if (count<nInputs) { 
				inputsWorking[iim*nInputs+count] = atof(cell.c_str()) ;
			} 
          count++;
        } // while
    } // for
    cout<<" Working set loaded. Inputs:"<<endl;
    data.close();
	return 0;
}



// reserves the memory and puts
//random values (inside min-max range) into weights  and biases
int NN::InitNet(double min, double max){
	
	cout<<" InitNet: nInputs="<<nInputs<<" nHiddenNeurons=";
	cout<<nHiddenNeurons<<" nOutputs="<<nOutputs<<endl; 
	// reserve the memory for weights and biases
	// hidden layer
	weightsHidden = new double[nHiddenNeurons*nInputs];
	biasHidden = new double[nHiddenNeurons];
	d_weightsHidden = new double[nHiddenNeurons*nInputs];
	d_biasHidden = new double[nHiddenNeurons];
	deltaHidden = new double[nHiddenNeurons];
	// output layer
	weightsOutput = new double[nHiddenNeurons*nOutputs];
	biasOutput = new double[nOutputs];
	d_weightsOutput = new double[nHiddenNeurons*nOutputs];
	d_biasOutput = new double[nOutputs];
	deltaOutput = new double[nOutputs];
	
	// current input and output vector, answers and error	
	currentInputs = new double[nInputs];
	currentOutputs = new double[nOutputs];
	currentAnswers = new double[nOutputs];
	currentError =  new double[nOutputs];
	
	// reserve memory for current net levels
	netHidden = new double[nHiddenNeurons];
	outHidden = new double[nHiddenNeurons];
	netOutput = new double[nOutputs];
	
	// make weights and biases random
	Randomize(weightsHidden,nHiddenNeurons*nInputs,min,max);
	Randomize(biasHidden,nHiddenNeurons,min,max);
	Randomize(weightsOutput,nHiddenNeurons*nOutputs,min,max);
	Randomize(biasOutput,nOutputs,min,max);
   
  	return 0;
}

// loads row of dataset into the net for estimation
void NN::GetTrainingEntry(int iTrainRow){
	for ( int i = 0 ; i<nInputs;i++)
	   currentInputs[i] = inputsTraining[iTrainRow*nInputs+i];
	for (int i = 0 ; i < nOutputs;i++)   
	  currentAnswers[i]= answersTraining[iTrainRow*nOutputs+i];
	
}

// loads row of dataset into the net for estimation
void NN::GetWorkingEntry(int iWorkRow){
	for ( int i = 0 ; i<nInputs;i++)
	   currentInputs[i] = inputsWorking[iWorkRow*nInputs+i];
	
}


// prints pixel values on the screen
void NN::DisplayTrainDigit(int iImage){
  int scan = 0;
  for (int i = 0 ; i < 8; i++){
    for ( int j = 0 ; j < 8;j++){
		//double a = inputsTraining[iImage*nInputs + scan];
		int colBand = (int)(inputsTraining[iImage*nInputs + scan]);
		cout<<setw(3)<<colBand<<" * ";
        scan++;
    }
    cout<<endl;
  }
}

void NN::DisplayWorkDigit(int iImage){
  int scan = 0;
  for (int i = 0 ; i < 8; i++){
    for ( int j = 0 ; j < 8;j++){
		//double a = inputsTraining[iImage*nInputs + scan];
		int colBand = (int)(inputsWorking[iImage*nInputs + scan]);
		cout<<setw(3)<<colBand<<" * ";
        scan++;
    }
    cout<<endl;
  }
}


// direct calculation of forward propagation
// takes input values and calculates the outputs
void NN::ForwardProp(){
	//  inputs ->  hidden layer
	// for each neuron in hidden layer
	for ( int hid = 0 ;hid < nHiddenNeurons ; hid++){
		// combine input values and add bias
        netHidden[hid] = biasHidden[hid]; 
        for (int inp = 0 ; inp < nInputs ; inp++){
		   netHidden[hid] = netHidden[hid] + 
		      currentInputs[inp]* weightsHidden[hid*nInputs + inp];
	    }
	    outHidden[hid] = activation(netHidden[hid]);
	}	
	sumOfOutputErrors = 0.0;
	
	// for each neuron in output layer
	for ( int out = 0 ; out < nOutputs ; out++){
		// combine hidden layer neuron outputs and add bias 
		netOutput[out] = biasOutput[out];
		for (int hid = 0 ; hid < nHiddenNeurons ; hid++){
			netOutput[out] = netOutput[out] +
			 outHidden[hid]* weightsOutput[out*nHiddenNeurons+hid];
		}
		currentOutputs[out] = activation(netOutput[out]);
		currentError[out] = currentOutputs[out] - currentAnswers[out];
		sumOfOutputErrors = sumOfOutputErrors + currentError[out]*currentError[out];
	}
}

// calculate gradient by direct estimation
void NN::DirectGradientEstimation(){
		double error0 = sumOfOutputErrors;
		/* hidden layer */
		for(int hidden=0; hidden<nHiddenNeurons; hidden++){
			//derviative for hidden layer bias
			biasHidden[hidden]+=dw;
			ForwardProp();
			double error1 = sumOfOutputErrors;
			d_biasHidden[hidden] = (error1-error0)/dw;
			biasHidden[hidden] -=dw;
		}
		for(int count=0; count<nInputs*nHiddenNeurons; count++){
				weightsHidden[count]+=dw;
				ForwardProp();
				double error1 = sumOfOutputErrors;
				d_weightsHidden[count] = (error1-error0)/dw;
				weightsHidden[count] -=dw;
			}
		/* output layer */
		for(int out=0; out<nOutputs; out++){
			//derivative for output bias values
			biasOutput[out]+=dw;
			ForwardProp();
			double error1 = sumOfOutputErrors;
			d_biasOutput[out] = (error1-error0)/dw;
			biasOutput[out] -=dw;
		}
		for(int count=0; count<nHiddenNeurons*nOutputs; count++){
				//derviavtive for output weights values
				weightsOutput[count]+=dw;
				ForwardProp();
				double error1 = sumOfOutputErrors;
				d_weightsOutput[count] = (error1-error0)/dw;
				weightsOutput[count] -=dw;
		}
}

// calculate delta by back-propagation for finding the derivatives
void NN::BackProp(){
		/* output layer */
		for(int out=0; out<nOutputs; out++){
			//finds the derivate for each neuron in the output layer
			deltaOutput[out] = d_activation(netOutput[out])*currentError[out];
		}
		/* hidden layer */
		for(int hidden=0; hidden<nHiddenNeurons; hidden++){
			//finds the derivate for each neuron in the output layer
			double value = 0;
			for(int out=0; out<nOutputs; out++){
				//sums up the output*weight for all outputs, using the weights from the current neuron
				value =value+ deltaOutput[out]*weightsOutput[hidden+nHiddenNeurons*out];
			}
			deltaHidden[hidden] = d_activation(netHidden[hidden])*value; //
		}
		stepBackProp();
}

//uses the delta values (calculated in BackProp() ) to find deriviatives for biases and weights
void NN::stepBackProp(){
	/* output layer */
		for(int out=0; out<nOutputs; out++){
			//find the derivate of each weight and bias in output layer
			d_biasOutput[out]=deltaOutput[out];
			for(int hidden=0; hidden<nHiddenNeurons; hidden++){
				//find derivative of weights
				d_weightsOutput[nHiddenNeurons*out+hidden]=deltaOutput[out]*outHidden[hidden];
			}
		}
		
		/* hidden layer */
		for(int hidden=0; hidden<nHiddenNeurons; hidden++){
			d_biasHidden[hidden] = deltaHidden[hidden]; //derivative of bias = delta
		}
		for(int input=0; input<nInputs; input++){
			for(int hidden=0; hidden<nHiddenNeurons; hidden++){
				d_weightsHidden[nInputs*hidden+input] = deltaHidden[hidden]*currentInputs[input]; //derivative of each weight = delta*first input
			}
		}	
}


// change weights and biases in direction oppposite to gradient,
// scaled by learning rate (which should be negative)
void NN::StepByGradient(){
		/* hidden layer */
		for(int count=0; count<nHiddenNeurons*nInputs; count++){
			//change weights of hidden layer
			weightsHidden[count] = weightsHidden[count] + learningRate*d_weightsHidden[count];
		}
		for(int hidden=0; hidden<nHiddenNeurons; hidden++){
			//derviative for hidden layer bias
			biasHidden[hidden] = biasHidden[hidden] + learningRate*d_biasHidden[hidden];
		}
		/* output layer */
		for(int out=0; out<nOutputs; out++){
			//cderivative for output bias values
			biasOutput[out] = biasOutput[out] + learningRate*d_biasOutput[out];
		}
		for(int hidden=0; hidden<nHiddenNeurons*nOutputs; hidden++){
			//derviavtive for output weights values
			weightsOutput[hidden] = weightsOutput[hidden] + learningRate*d_weightsOutput[hidden];

		}
}


// calculates combined error for all entries in the dataset
// for current values of weights and biases
double NN::TotalDatasetError(){ // sum of errors for all rows of train data
	totalDatasetError = 0.0;
	for ( int entry = 0 ; entry < nTrainingEntries; entry++){
		GetTrainingEntry(entry);
	    ForwardProp();
	    totalDatasetError = totalDatasetError + GetOutputError();
	}
	return totalDatasetError;
}

// tunes/trains the neural network
void NN::Train1(){
	// set net search parameters
	learningRate = -0.5;
    //DisplayTrainDigit(iImage);
    InitNet(-0.1,0.1);
    int iImage = 0;
    srand (time(NULL));  // seed random number generator
    int searchStep = 0;
    
    while (( searchStep < 15000) && (TotalDatasetError() > 10.0) ){ 
  	  // pick random entry from training dataset
      iImage = nTrainingEntries*(double)rand() / RAND_MAX;
	  // copy inputs and outputs from training matrix into neural net
	  //learningRate = -0.15-0.5*pow(1-0.003, searchStep);
  	  GetTrainingEntry(iImage);
      ForwardProp();
      //DirectGradientEstimation();
      BackProp();
      StepByGradient();
      //stepBackProp();
      cout<<"step: "<<searchStep<<" image: "<<iImage<<" Error for current row:"<<GetOutputError();
      cout<<" Total dataset error: "<< TotalDatasetError()<<" Learning rate= "<<learningRate<<endl;
      searchStep++;
    }
     
}

void NN::PrintOutputs(){
	cout<<" Net outputs: ";
	for (int out = 0 ; out < nOutputs ; out++){
		cout<<currentOutputs[out]<<"  ";
	}
	cout<<endl;
}


void NN::DisplayResults(){
	int iImage = -1;
	cout<<" There are "<< nTrainingEntries<<" entries "<<endl;
	cout<<" Enter number of the entry to display"<<endl;
	cin>>iImage;
	while (iImage < nTrainingEntries) {
	  // copy inputs and outputs from big matrix
	  GetTrainingEntry(iImage);
	  ForwardProp();
      DisplayTrainDigit(iImage);
      PrintVector(currentOutputs, nOutputs);
      cin>>iImage;
   }
    
}


int main(){
	NN neuralNet;
	neuralNet.LoadTrainingSet("train.txt",64,128,8);
	//neuralNet.DisplayTrainDigit(0);
	int imageIndex = 0;
	std::cout<<"Enter index of the image to display (>750 to continue)"<<std::endl;
	std::cin>>imageIndex;
    neuralNet.DisplayTrainDigit(imageIndex);
	while ( imageIndex < 750){
	  cout<<" Enter index of the imag to display"; cin>>imageIndex;
	  if (imageIndex<759){
		   neuralNet.DisplayTrainDigit(imageIndex);
	  }
    }
	//int imageIndex = 0 ;
	neuralNet.InitNet(-0.1,0.1);
	neuralNet.ForwardProp();
	neuralNet.Train1();
	
	imageIndex = 0;
	std::cout<<"Enter index of the image to display"<<std::endl;
	std::cin>>imageIndex;
	neuralNet.DisplayTrainDigit(imageIndex);
	while ( imageIndex < 750){
	  cout<<" Enter index of the imag to display"; cin>>imageIndex;
	  if (imageIndex<759){
		   neuralNet.GetTrainingEntry(imageIndex);
	       neuralNet.ForwardProp();
    	   neuralNet.DisplayTrainDigit(imageIndex);
	       neuralNet.PrintOutputs();
      }
    }
    neuralNet.LoadWorkingSet("work.txt",64);
	std::cin>>imageIndex;
	neuralNet.DisplayWorkDigit(imageIndex);
	while ( imageIndex < 750){
	  cout<<" Enter index of the imag to display"; cin>>imageIndex;
	  if (imageIndex<759){
		   neuralNet.GetWorkingEntry(imageIndex);
	       neuralNet.ForwardProp();
    	   neuralNet.DisplayWorkDigit(imageIndex);
	       neuralNet.PrintOutputs();
      }
    }
    
}
