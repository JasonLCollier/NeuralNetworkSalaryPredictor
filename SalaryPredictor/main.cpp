#include "header.hpp"
#include "functions.cpp"

int main() {
	double tp = 0; //target output
	double op = 0; //actual output
	double n = 0.000005; //learning rate
	int p = 0; //input pattern number
	double SSE0 = 0; //previous sum of squared errors
	double SSE1 = std::numeric_limits<double>::max(); //current sum of squared errors
	int testData[2000][9]; //training set of 2000 patterns(p)
	double v[8]; //weight array
	int evalSetSize = 10; //size of evaluation set
	double duration; //Exexution time
	std::clock_t start;

	//Read the SalData.csv file
	std::ifstream salDataFile("SalData.csv");
	std::string row, col;
	if (salDataFile.is_open())
	{
		int  r = 0;
		std::getline(salDataFile, row); //skip first line
		while (std::getline(salDataFile, row))
		{
			std::stringstream ssRow(row);
			int c = 0;
			while (std::getline(ssRow, col, ',')) {
				std::stringstream ssCol(col);
				ssCol >> testData[r][c]; //populate the input signals(zi)
				c++;
			}
			testData[r][8] = -1; //populate the bias
			r++;
		}
		salDataFile.close();
	}

	for (int i = 0; i < 8; i++) {
		v[i] = 0.5; //populate weights matrix with initial weights
	}

	start = std::clock();
	int iterations = 0;
	do//keep iterating while not successful
	{
		SSE0 = SSE1;
		SSE1 = 0;
		for (int P = evalSetSize; P < 2000; P++)
		{
			tp = testData[P][0];
			op = 0;
			for (int I = 0; I < 8; I++)
				op += testData[P][I + 1] * v[I]; //calculate op using linear activation function
			for (int I = 0; I < 8; I++)
			{
				v[I] -= n * (-2 * (tp - op) * testData[P][I + 1]); //update weights using Widrow Hoff learning rule (least means square algortithm)
			}
			SSE1 += pow((tp - op), 2); //Update the SSE (Gradient Descent Learning Rule)
		}
		iterations++;
	} while (SSE1 < SSE0);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	std::cout << "Training set size: " << 2000 - evalSetSize << std::endl;
	std::cout << "Training Time: " << duration << std::endl;
	std::cout << "Iterations for training: " << iterations << std::endl;
	std::cout << "Weights: " << std::endl;
	for (int i = 0; i < 8; i++)
	{
		std::cout<< v[i] << std::endl;
	}

	std::cout << "\nValidation set size: " << evalSetSize << std::endl;
	std::cout << "Target Value\t\tActual Output" << std::endl;
	double targetSum = 0;
	double actualSum = 0;
	for (int P = 0; P < evalSetSize; P++)
	{
		double calcVal = 0;
		for (int I = 0; I < 8; I++)
		{
			calcVal += v[I] * testData[P][I + 1];
		}
		std::cout << testData[P][0] << "\t\t\t" << calcVal << std::endl;
		targetSum += testData[P][0];
		actualSum += calcVal;
	}
	std::cout << "Overall error percentage: " << abs(1 - targetSum / actualSum) * 100 << "%" << std::endl;

	std::string str;
	getline(std::cin, str);
	return 0;
}

