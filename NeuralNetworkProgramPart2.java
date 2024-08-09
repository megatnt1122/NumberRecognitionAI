import java.lang.Math;
import java.util.Scanner;
import java.io.*;
import java.util.Random;


public class NeuralNetworkProgramPart2 
{
	public static void main(String[] args) throws IOException 
	{
		int isMode7Availiable = 0;
		int isModeAvailiable = 0;
		int mode = 0;
		
		//MAKE SURE TO INCLUDE THE \\output.txt OTHERWISE IT WILL ERROR OUT *********************************
		String inputFilePath = "E:\\ProfilingStuff\\finalreport\\NNP2AustinHampton\\mnist_train.csv";
		String testingInputFilePath = "E:\\ProfilingStuff\\finalreport\\NNP2AustinHampton\\mnist_test.csv";
		String hiddenWeightFilePath = "E:\\ProfilingStuff\\finalreport\\NNP2AustinHampton\\RecordedHiddenWeights.csv";
		String outputWeightFilePath = "E:\\ProfilingStuff\\finalreport\\NNP2AustinHampton\\RecordedOutputWeights.csv";
		String csvHiddenWeightOutput = "";
		String csvOutputWeightOutput = "";
		int numOfEpochs = 1;
		int numOfMinibatches = 6000;
		int numOfTrainingSetsPerMinibatch = 10;
		int numOfNodesInInputLayer = 784;
		int numOfNodesInHiddenLayer = 30;
		int numOfNodesInOutputLayer = 10;
		
		double[][] recordedHiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
		double[][] recordedOutputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
		double[] recordedHiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
		double[] recordedOutputLayerBiasVector = new double[numOfNodesInOutputLayer];
		/*
		TO DO:
		
			First we need to convert the system to take in the MNIST training data in a way it can read
			
			SYNTAX IS:
			The Label is the desired output while the pix's is the image itself
			
			NOTE: That the pixels are greyscale from 0-255. 0 being black and 255 being white
			LABEL(0-9), PIX-X1-Y1, PIX-X1-Y2, PIX-X1-Y3, PIX-X28-Y28
							
					(2) For best results, scale your pixel inputs to be a fraction between 0 and 1 by dividing every pixel input by 255, storing the result as a
						double.
		
		*/
		Scanner scan = new Scanner(System.in);
		while(true)
		{
		
			mode = 0;
			//SCAN TO SAY WHICH MODE WE WAINT
			System.out.println("Select one below: ");
			System.out.println("[1] Train the network");
			System.out.println("[2] Load a pre-trained network");
			System.out.println("[3] Display network accuracy on TRAINING data");
			System.out.println("[4] Display network accuracy on TESTING data");
			System.out.println("[5] Run network on TESTING data showing images and labels");
			System.out.println("[6] Display the misclassified TESTING images");
			System.out.println("[7] Save the network state to file");
			System.out.println("[0] Exit");
			
			mode = scan.nextInt();
if(mode == 1)
			{
				numOfEpochs = 30;
				numOfMinibatches = 6000;
				//Converts CSV To an dimensional Array (NOTE): The reason the dimension adds a 1 is because the label is at the front of the input set
				double[][]  convertedInputSets = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer + 1];
				convertedInputSets = convertCsvToArray(inputFilePath, numOfMinibatches, numOfTrainingSetsPerMinibatch, numOfNodesInInputLayer);
				
				//Randomly shuffled inputsets
				convertedInputSets = shuffleArray(convertedInputSets);
				
				
				 
						

				
				//This is also how many training sets we want per minibatch
				double[][]  hiddenLayerInputVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer];
				hiddenLayerInputVector = inputVectorGrabber(convertedInputSets);
				
				//array of the hidden layers initial weight vector
				double[][]  hiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				hiddenLayerWeightVector = hiddenLayerWeightVectorGrabber(numOfNodesInHiddenLayer, numOfNodesInInputLayer);
				
				//Empty revisingWeightVector that will be used to store the revised weights before cloning them to the original array to replace the old weights
				double[][] revisedHiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				
				//Array that is the initial biases manually set up using our test inputs... will be changed to random later
				double[] hiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				hiddenLayerBiasVector = hiddenLayerBiasVectorGrabber(numOfNodesInHiddenLayer);
				
				//Empty array used to store the revised hidden layer bias vector for later use
				double[] revisedHiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				
				//Array of the output layers initial weight vector
				double[][] outputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
				outputLayerWeightVector = outputLayerWeightVectorGrabber(numOfNodesInOutputLayer, numOfNodesInHiddenLayer);
				
				//Empty array that will be used to store the revised output layer weight vector
				double[][] revisedOutputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
			
				//Array of the output layers initial biases
				double[] outputLayerBiasVector = new double[numOfNodesInOutputLayer];
				outputLayerBiasVector = outputLayerBiasVectorGrabber(numOfNodesInOutputLayer);
				
				double[] revisedOutputLayerBiasVector = new double[numOfNodesInOutputLayer];
				
				//Array of the output layers desired outputs for training
				double[][] outputLayerTrainingVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				outputLayerTrainingVector = outputLayerTrainingVectorGrabber(numOfNodesInOutputLayer, numOfMinibatches,numOfTrainingSetsPerMinibatch, convertedInputSets);
			
				
				//Compensator for cases in which we have multiple training sets per minibatch
				int miniBatchCompensator = 0;
				
				int zMiniBatchCompensator = 0;
				
				//Array containing the current training sets
				double[][] currentHiddenLayerZSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInHiddenLayer];
				
				//This only intakes 10 of the 60000 sets we have so we situate this to randomly choose 10 input sets
				//Initializes an array we will use as our current input set
				double[] currentHiddenLayerInputSet = new double[numOfNodesInInputLayer];
				
				//This is used to temporarily store the hidden layer activation sums
				double[][] currentHiddenLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//This is used to temporarily store the current output layer Z
				double[][] currentOutputLayerZSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Initializes an array we will use as our current input set
				double[] currentOutputLayerInputSet = new double[numOfNodesInOutputLayer];
			
				//This is used to temporarily store the output layer activation sums
				double[][] currentOutputLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Temporarily grabs the currently used desired output from the premade array
				double[][] currentOutputLayerTrainingSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//Temporarily stores the output of the bias gradients for that pass to use later in the loop
				double[][] currentOutputLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//DIMENSIONS
				//Temporarily stores the output of the weight gradient once everything has been thrown through the equations
				double[][][] currentOutputLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer][numOfNodesInHiddenLayer];	
			
				//Temporarily stores the output of the hidden Layer bias gradient functions for use in the loop
				double[][] currentHiddenLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//Temporarily stores the output of the hidden Layer weight gradient for use in the loop and revision
				double[][][] currentHiddenLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length][hiddenLayerInputVector.length];
				
				//System.out.println(hiddenLayerInputVector[6000]);
				
				
				
				
				
				
				
				
				
				
				
				
				//THIS IS FOR FUTURE AUSTIN... THERE IS AN ISSUE... WE NEED THE LOOPS TO DO 10 TRAINING SETS THEN CHANGE MINIBATCHES... SO 0,1,2,3,4,5,6,7,8,9... 0,1,2,3... etc etc
				//The issue is that the zCompensator is running after only 2 training sets which is wrong and throws it to become 10 it needs to only change to 10 ONCE we looped through
				//all the initial training sets first so I DO 10 SETS then say ok make my zCompensator 10 etc etc
				//
				
				
				
				
				
				
				
				
				
				
				
				
				
				//Counts total number of digits
				int totalZeros = 0;
				int totalOnes = 0;
				int totalTwos = 0;
				int totalThrees = 0;
				int totalFours = 0;
				int totalFives = 0;
				int totalSixs = 0;
				int totalSevens = 0;
				int totalEights = 0;
				int totalNines = 0;
				
				//Counts total number of correct digits
				int totalCorrectlyGuessedZeros = 0;
				int totalCorrectlyGuessedOnes = 0;
				int totalCorrectlyGuessedTwos = 0;
				int totalCorrectlyGuessedThrees = 0;
				int totalCorrectlyGuessedFours = 0;
				int totalCorrectlyGuessedFives = 0;
				int totalCorrectlyGuessedSixs = 0;
				int totalCorrectlyGuessedSevens = 0;
				int totalCorrectlyGuessedEights = 0;
				int totalCorrectlyGuessedNines = 0;
				
				int totalCorrectlyGuessedDigits = 0;
				double Accuracy = 0;
				int totalDigits = 0;
				
				//Checks if 0 is the current number
				int desiredIsZero = 0;
				
				double maxElementIndex = -1;
				
			
				//BIG BOY LOOP THAT DOES IT ALL
				//LOOP FOR EPOCHS 1
				for (int i = 0; i < numOfEpochs; i++)
				{

					//RESETTING THE COUNTS
					 totalZeros = 0;
					 totalOnes = 0;
					 totalTwos = 0;
					 totalThrees = 0;
					 totalFours = 0;
					 totalFives = 0;
					 totalSixs = 0;
					 totalSevens = 0;
					 totalEights = 0;
					 totalNines = 0;
					
					//RESETTING THE COUNTS
					 totalCorrectlyGuessedZeros = 0;
					 totalCorrectlyGuessedOnes = 0;
					 totalCorrectlyGuessedTwos = 0;
					 totalCorrectlyGuessedThrees = 0;
					 totalCorrectlyGuessedFours = 0;
					 totalCorrectlyGuessedFives = 0;
					 totalCorrectlyGuessedSixs = 0;
					 totalCorrectlyGuessedSevens = 0;
					 totalCorrectlyGuessedEights = 0;
					 totalCorrectlyGuessedNines = 0;
					 
					 totalCorrectlyGuessedDigits = 0;
					 Accuracy = 0;
					 totalDigits = 0;
					 
					 miniBatchCompensator = 0;
					 maxElementIndex = -1;
					 
					//loop for each minibatch
					for (int j = 0; j < numOfMinibatches; j++)
					{
						//System.out.println("I am within the outer loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//loop for each training loop
						//System.out.println("K should have gone to 10");
						//System.out.println("Comp is : " + miniBatchCompensator);
						
						
						//ERROR: THIS LOOP IS GOING 6000 TIMES... which makes sense it should but the loop below isnt incrementing that at all.... 
						//			essientially its only counting 10 digits then saying eh who cares... FIX THE MINI BATCH COMPENSATOR IS 0 INITIALLY THIS IS WHY
						
						
						for (int k = miniBatchCompensator; k < numOfTrainingSetsPerMinibatch + miniBatchCompensator; k++)
						{
			
			
					
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				FORWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
							// TODO Auto-generated method stub
							//1. Sigmoid activation function = 1/(1+e^(-z))
							//		---e is the Eulers number = 2.71828
							//		---z is the DOT PRODUCT of the vector W with vector X plus the bias
							//		---vector W is a vector containing all the weights
							//		---vector X is a vector containing all the inputs
							//		---bias is the assigned bias
							//
							//		To do this we need to first have a way to set the weights and inputs
							//		
							//		Then we can feed those vectors into a method that finds the Z
							//
							//		Then we can insert Z into the SAF in order to spit out the activation value
							// 		
							//		Then we run the results back through
							
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							//System.out.println(zMiniBatchCompensator);
							//sets the first input set and uses it below
							
							currentHiddenLayerInputSet =  hiddenLayerInputVector[k].clone();
							
						
							//System.out.println("I am within the INNER loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//****CURRENT ERROR IS HERE SOMETHING GOES OUT OF BOUNDS OF 10 *********
							//currentHiddenLayerZSets uses the zMiniBatchCompensator because k is situated in a way to go 0 1 2 3... while the max zsets we can hold are only 0 1
							currentHiddenLayerZSets[zMiniBatchCompensator] = zAssignment(hiddenLayerWeightVector, currentHiddenLayerInputSet, hiddenLayerBiasVector);
							
							//Assigns the outcome of throwing the currentHiddenLayerZSets through the sigmoidal function
							currentHiddenLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentHiddenLayerZSets[zMiniBatchCompensator]);
					
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Grabs the current input set we acquired from the activation sums of the previous layer
							currentOutputLayerInputSet = currentHiddenLayerActivationSumSets[zMiniBatchCompensator].clone();
							
							//Throws the new input set through the zAssignment function and grabs the output layer Z sets from the equation
							currentOutputLayerZSets[zMiniBatchCompensator] = zAssignment(outputLayerWeightVector, currentOutputLayerInputSet, outputLayerBiasVector);
												
							//Temporarily stores the output layer activation sums after going through the sigmoidal function
							currentOutputLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentOutputLayerZSets[zMiniBatchCompensator]);
			
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				BACKWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							
							//Next is to create the back propagation things that HAVE SEPERATE EQUATIONS FOR INTENRAL AND EXTERNAL LAYERS !!!!!
							//The equation for the error/ THE BIASGRADIENT in the output layer is L = (a-y) * a * (1-a) 
							// a being the outputLayerActivationSum
							//y being the desired activation value
							
							//Method that grabs the current desired input sets to use later within the loop
							currentOutputLayerTrainingSets[zMiniBatchCompensator] = outputLayerTrainingVector[k].clone();
							
							//SO THE TABLE SHOWS A CONSISTENT PATTERN WHICH ISNT GOOD IT SHOULD BE COMPLETELY RANDOM...
							//Increments the corresponding numbers aka if this is the correct output and the total
							
							//Max element in the list
							//System.out.println("Zmini is " + zMiniBatchCompensator);
							maxElementIndex = maxElementInArray(currentOutputLayerActivationSumSets[zMiniBatchCompensator]);
							desiredIsZero = 1;
							
							//If its desired is 1 then increment total, then if guess is max of the set then also increment it
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][1] == 1.0)
							{
								totalOnes = 1 + totalOnes;
								if (maxElementIndex == 1)
								{
									totalCorrectlyGuessedOnes = 1 + totalCorrectlyGuessedOnes;
								}
								desiredIsZero = 0;
							}
							
							else if(currentOutputLayerTrainingSets[zMiniBatchCompensator][2] == 2.0)
							{
								totalTwos = 1 + totalTwos;
								if(maxElementIndex == 2)
								{
									totalCorrectlyGuessedTwos = 1 + totalCorrectlyGuessedTwos;
								}
								desiredIsZero = 0;
							}
							
							else if(currentOutputLayerTrainingSets[zMiniBatchCompensator][3] == 3.0)
							{
								totalThrees = 1 + totalThrees;
								if(maxElementIndex == 3)
								{
									totalCorrectlyGuessedThrees = 1 + totalCorrectlyGuessedThrees;
								}
								desiredIsZero = 0;
							}
							
							else if(currentOutputLayerTrainingSets[zMiniBatchCompensator][4] == 4.0)
							{
								totalFours = 1 + totalFours;
								if(maxElementIndex == 4)
								{
									totalCorrectlyGuessedFours = 1 + totalCorrectlyGuessedFours;
								}
								desiredIsZero = 0;
							}
							
							else if(currentOutputLayerTrainingSets[zMiniBatchCompensator][5] == 5.0)
							{
								totalFives = 1 + totalFives;
								if(maxElementIndex == 5)
								{
									totalCorrectlyGuessedFives = 1 + totalCorrectlyGuessedFives;
								}
								desiredIsZero = 0;
							}
							
							else if(currentOutputLayerTrainingSets[zMiniBatchCompensator][6] == 6.0)
							{
								totalSixs = 1 + totalSixs;
								if(maxElementIndex == 6)
								{
									totalCorrectlyGuessedSixs = 1 + totalCorrectlyGuessedSixs;
								}
								desiredIsZero = 0;
							}
							
							else if(currentOutputLayerTrainingSets[zMiniBatchCompensator][7] == 7.0)
							{
								totalSevens = 1 + totalSevens;
								if(maxElementIndex == 7)
								{
									totalCorrectlyGuessedSevens = 1 + totalCorrectlyGuessedSevens;
								}
								desiredIsZero = 0;
							}
							
							else if(currentOutputLayerTrainingSets[zMiniBatchCompensator][8] == 8.0)
							{
								totalEights = 1 + totalEights;
								if(maxElementIndex == 8)
								{
									totalCorrectlyGuessedEights = 1 + totalCorrectlyGuessedEights;
								}
								desiredIsZero = 0;
							}
							
							else if(currentOutputLayerTrainingSets[zMiniBatchCompensator][9] == 9.0)
							{
								totalNines = 1 + totalNines;
								if(maxElementIndex == 9)
								{
									totalCorrectlyGuessedNines= 1 + totalCorrectlyGuessedNines;
								}
								desiredIsZero = 0;
							}
							
							else
							{
								totalZeros = 1 + totalZeros;
								if(maxElementIndex == 0)
								{
									totalCorrectlyGuessedZeros = 1 + totalCorrectlyGuessedZeros;
								}
							}
							
							maxElementIndex = -1;
									
						
							//Method that temporarily grabs the bias gradients for later use in the loops
							currentOutputLayerBiasGradientSets[zMiniBatchCompensator] = outputLayerBiasGradientFunction(currentOutputLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerTrainingSets[zMiniBatchCompensator]);
							
							//After the bias gradient we would need to then find the outputWeightGradient which is the previous's layer activation sum * the gradient bias
							currentOutputLayerWeightGradientSets[zMiniBatchCompensator] = outputLayerWeightGradientFunction(currentOutputLayerBiasGradientSets[zMiniBatchCompensator], currentHiddenLayerActivationSumSets[zMiniBatchCompensator]);
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Temporarily stores the current hidden layer bias gradient for later use in revising the biases
							currentHiddenLayerBiasGradientSets[zMiniBatchCompensator] = hiddenLayerBiasGradientFunction( outputLayerWeightVector, currentHiddenLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//Gathers the hidden layers weight gradient to use in the future		
							currentHiddenLayerWeightGradientSets[zMiniBatchCompensator] = hiddenLayerWeightGradientFunction(currentHiddenLayerInputSet, currentHiddenLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//used to alternate between the 10 training sets that we use
							zMiniBatchCompensator = 1 + zMiniBatchCompensator;
						}
						zMiniBatchCompensator = 0;
						
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						//REVISING BIASES AND WEIGHTS          						    															  //
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						
						//Revised weights for the hidden layer TILES BF26 TO BI30
						revisedHiddenLayerWeightVector = RevisedWeightVectorFunction(currentHiddenLayerWeightGradientSets, hiddenLayerWeightVector);
						
						//Revised weights for the output layer TILES BF35 TO BH36
						revisedOutputLayerWeightVector = RevisedWeightVectorFunction(currentOutputLayerWeightGradientSets, outputLayerWeightVector);
						
						//Revised biases for the hidden layer tiles BD27 to BD29
						revisedHiddenLayerBiasVector = RevisedBiasVectorFunction(currentHiddenLayerBiasGradientSets, hiddenLayerBiasVector); 
						
						//Revised biases for the output layer tiles BD35 to BD36
						//double[] outputLayerRevisedBiasVector = RevisedBiasVectorFunction(outputLayerBiasGradientTrainingCase2, outputLayerBiasGradient, outputLayerBiasVector);
						revisedOutputLayerBiasVector = RevisedBiasVectorFunction(currentOutputLayerBiasGradientSets, outputLayerBiasVector);
						
						
						
						
			
						
						
						
						
						
						
						
						
						
						
						//SETTING THE NEW WEIGHTS AND BIASES
						
						//setting new revised hidden layer weights
						hiddenLayerWeightVector = revisedHiddenLayerWeightVector.clone();
						
						//setting new revised output layer weights
						outputLayerWeightVector = revisedOutputLayerWeightVector.clone();
						
						//setting new revised hidden layer biases
						hiddenLayerBiasVector = revisedHiddenLayerBiasVector.clone();
						
						//setting new revised output layer biases
						outputLayerBiasVector = revisedOutputLayerBiasVector.clone();
						
						//Used to alternate between each training set if theres 2
						miniBatchCompensator = numOfTrainingSetsPerMinibatch + miniBatchCompensator;
						//System.out.println("REVISEDminiBatchCompensator = " + miniBatchCompensator);
					}
					
					//Storing the final weights and biases to be used later
					recordedHiddenLayerWeightVector = hiddenLayerWeightVector.clone();
					recordedOutputLayerWeightVector = outputLayerWeightVector.clone();
					recordedHiddenLayerBiasVector = hiddenLayerBiasVector.clone();
					recordedOutputLayerBiasVector = outputLayerBiasVector.clone();
					
					totalCorrectlyGuessedDigits = totalCorrectlyGuessedZeros+totalCorrectlyGuessedOnes+totalCorrectlyGuessedTwos+totalCorrectlyGuessedThrees+totalCorrectlyGuessedFours+totalCorrectlyGuessedFives+totalCorrectlyGuessedSixs+totalCorrectlyGuessedSevens+totalCorrectlyGuessedEights+totalCorrectlyGuessedNines;
					totalDigits = totalZeros + totalOnes + totalTwos + totalThrees + totalFours + totalFives + totalSixs + totalSevens + totalEights + totalNines;
					Accuracy = (Double.valueOf(totalCorrectlyGuessedDigits)/Double.valueOf(totalDigits));
					//OUTPUT:
					System.out.println("Epoch: " + i);
					System.out.println("	0: "+ totalCorrectlyGuessedZeros+"/"+ totalZeros);
					System.out.println("	1: " + totalCorrectlyGuessedOnes +"/"+ totalOnes);
					System.out.println("	2: "+ totalCorrectlyGuessedTwos +"/"+ totalTwos);
					System.out.println("	3: "+ totalCorrectlyGuessedThrees +"/"+ totalThrees);
					System.out.println("	4: "+ totalCorrectlyGuessedFours +"/"+ totalFours);
					System.out.println("	5: "+ totalCorrectlyGuessedFives +"/"+ totalFives);
					System.out.println("	6: "+ totalCorrectlyGuessedSixs +"/"+ totalSixs);
					System.out.println("	7: "+ totalCorrectlyGuessedSevens +"/"+ totalSevens);
					System.out.println("	8: "+ totalCorrectlyGuessedEights +"/"+ totalEights);
					System.out.println("	9: "+ totalCorrectlyGuessedNines +"/"+ totalNines);
					System.out.println("	Accuracy = "+ totalCorrectlyGuessedDigits +"/" + totalDigits + " = " + Accuracy*100 + "%");
					
				}
				
				isMode7Availiable = 1;
				isModeAvailiable = 1;
			}
			
			//Mode 2 is to load weights and biases into the network
			if(mode == 2)
			{
				System.out.println("Loading pretrained network");
				
				//converting the stored files inot the appropriate variables
				recordedHiddenLayerWeightVector = convertCsvToHiddenWeight(hiddenWeightFilePath,numOfNodesInHiddenLayer, numOfNodesInInputLayer);
				recordedOutputLayerWeightVector = convertCsvToOutputWeight(outputWeightFilePath, numOfNodesInOutputLayer, numOfNodesInHiddenLayer);
					
				isModeAvailiable = 1;
			}
			
//ONLY HAS A SINGLE EPOCH AND RUNS THROUGH THE TRAINING SET ONCE USING THE LOADED RECORDED WEIGHT DATA
if(mode == 3)
			{
	if(isModeAvailiable == 0)
	{
		System.out.println("You have not selected any weights yet...");
	}
	else
	{
		
				numOfEpochs = 1;
				numOfMinibatches = 6000;
				//setting new revised hidden layer weights
				//Converts CSV To an dimensional Array (NOTE): The reason the dimension adds a 1 is because the label is at the front of the input set
				double[][]  convertedInputSets = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer + 1];
				convertedInputSets = convertCsvToArray(inputFilePath, numOfMinibatches, numOfTrainingSetsPerMinibatch, numOfNodesInInputLayer);
				
				//Randomly shuffled inputsets
				convertedInputSets = shuffleArray(convertedInputSets);
				
				
				 

				
				//This is also how many training sets we want per minibatch
				double[][]  hiddenLayerInputVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer];
				hiddenLayerInputVector = inputVectorGrabber(convertedInputSets);
				
				//array of the hidden layers initial weight vector
				double[][]  hiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				hiddenLayerWeightVector = recordedHiddenLayerWeightVector.clone();
				
				//Empty revisingWeightVector that will be used to store the revised weights before cloning them to the original array to replace the old weights
				double[][] revisedHiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				
				//Array that is the initial biases manually set up using our test inputs... will be changed to random later
				double[] hiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				hiddenLayerBiasVector = hiddenLayerBiasVectorGrabber(numOfNodesInHiddenLayer);
				
				//Empty array used to store the revised hidden layer bias vector for later use
				double[] revisedHiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				
				//Array of the output layers initial weight vector
				double[][] outputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
				outputLayerWeightVector = recordedOutputLayerWeightVector.clone();
				
				//Empty array that will be used to store the revised output layer weight vector
				double[][] revisedOutputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
			
				//Array of the output layers initial biases
				double[] outputLayerBiasVector = new double[numOfNodesInOutputLayer];
				outputLayerBiasVector = outputLayerBiasVectorGrabber(numOfNodesInOutputLayer);
				
				double[] revisedOutputLayerBiasVector = new double[numOfNodesInOutputLayer];
				
				//Array of the output layers desired outputs for training
				double[][] outputLayerTrainingVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				outputLayerTrainingVector = outputLayerTrainingVectorGrabber(numOfNodesInOutputLayer, numOfMinibatches,numOfTrainingSetsPerMinibatch, convertedInputSets);
			
				
				//Compensator for cases in which we have multiple training sets per minibatch
				int miniBatchCompensator = 0;
				
				int zMiniBatchCompensator = 0;
				
				//Array containing the current training sets
				double[][] currentHiddenLayerZSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInHiddenLayer];
				
				//This only intakes 10 of the 60000 sets we have so we situate this to randomly choose 10 input sets
				//Initializes an array we will use as our current input set
				double[] currentHiddenLayerInputSet = new double[numOfNodesInInputLayer];
				
				//This is used to temporarily store the hidden layer activation sums
				double[][] currentHiddenLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//This is used to temporarily store the current output layer Z
				double[][] currentOutputLayerZSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Initializes an array we will use as our current input set
				double[] currentOutputLayerInputSet = new double[numOfNodesInOutputLayer];
			
				//This is used to temporarily store the output layer activation sums
				double[][] currentOutputLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Temporarily grabs the currently used desired output from the premade array
				double[][] currentOutputLayerTrainingSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//Temporarily stores the output of the bias gradients for that pass to use later in the loop
				double[][] currentOutputLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//DIMENSIONS
				//Temporarily stores the output of the weight gradient once everything has been thrown through the equations
				double[][][] currentOutputLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer][numOfNodesInHiddenLayer];	
			
				//Temporarily stores the output of the hidden Layer bias gradient functions for use in the loop
				double[][] currentHiddenLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//Temporarily stores the output of the hidden Layer weight gradient for use in the loop and revision
				double[][][] currentHiddenLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length][hiddenLayerInputVector.length];
				
				//System.out.println(hiddenLayerInputVector[6000]);
				
				
				
				
				
				
				
				
				
				
				
				
				//THIS IS FOR FUTURE AUSTIN... THERE IS AN ISSUE... WE NEED THE LOOPS TO DO 10 TRAINING SETS THEN CHANGE MINIBATCHES... SO 0,1,2,3,4,5,6,7,8,9... 0,1,2,3... etc etc
				//The issue is that the zCompensator is running after only 2 training sets which is wrong and throws it to become 10 it needs to only change to 10 ONCE we looped through
				//all the initial training sets first so I DO 10 SETS then say ok make my zCompensator 10 etc etc
				//
				
				
				
				
				
				
				
				
				
				
				
				
				
				//Counts total number of digits
				int totalZeros = 0;
				int totalOnes = 0;
				int totalTwos = 0;
				int totalThrees = 0;
				int totalFours = 0;
				int totalFives = 0;
				int totalSixs = 0;
				int totalSevens = 0;
				int totalEights = 0;
				int totalNines = 0;
				
				//Counts total number of correct digits
				int totalCorrectlyGuessedZeros = 0;
				int totalCorrectlyGuessedOnes = 0;
				int totalCorrectlyGuessedTwos = 0;
				int totalCorrectlyGuessedThrees = 0;
				int totalCorrectlyGuessedFours = 0;
				int totalCorrectlyGuessedFives = 0;
				int totalCorrectlyGuessedSixs = 0;
				int totalCorrectlyGuessedSevens = 0;
				int totalCorrectlyGuessedEights = 0;
				int totalCorrectlyGuessedNines = 0;
				
				int totalCorrectlyGuessedDigits = 0;
				double Accuracy = 0;
				int totalDigits = 0;
				
				//Checks if 0 is the current number
				int desiredIsZero = 0;
				
				double maxElementIndex = -1;
				
			
				//BIG BOY LOOP THAT DOES IT ALL
				//LOOP FOR EPOCHS 1
				for (int i = 0; i < numOfEpochs; i++)
				{

					//RESETTING THE COUNTS
					 totalZeros = 0;
					 totalOnes = 0;
					 totalTwos = 0;
					 totalThrees = 0;
					 totalFours = 0;
					 totalFives = 0;
					 totalSixs = 0;
					 totalSevens = 0;
					 totalEights = 0;
					 totalNines = 0;
					
					//RESETTING THE COUNTS
					 totalCorrectlyGuessedZeros = 0;
					 totalCorrectlyGuessedOnes = 0;
					 totalCorrectlyGuessedTwos = 0;
					 totalCorrectlyGuessedThrees = 0;
					 totalCorrectlyGuessedFours = 0;
					 totalCorrectlyGuessedFives = 0;
					 totalCorrectlyGuessedSixs = 0;
					 totalCorrectlyGuessedSevens = 0;
					 totalCorrectlyGuessedEights = 0;
					 totalCorrectlyGuessedNines = 0;
					 
					 totalCorrectlyGuessedDigits = 0;
					 Accuracy = 0;
					 totalDigits = 0;
					 
					 miniBatchCompensator = 0;
					 maxElementIndex = -1;
					 
					//loop for each minibatch
					for (int j = 0; j < numOfMinibatches; j++)
					{
						//System.out.println("I am within the outer loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//loop for each training loop
						//System.out.println("K should have gone to 10");
						//System.out.println("Comp is : " + miniBatchCompensator);
						
						
						//ERROR: THIS LOOP IS GOING 6000 TIMES... which makes sense it should but the loop below isnt incrementing that at all.... 
						//			essientially its only counting 10 digits then saying eh who cares... FIX THE MINI BATCH COMPENSATOR IS 0 INITIALLY THIS IS WHY
						
						
						for (int k = miniBatchCompensator; k < numOfTrainingSetsPerMinibatch + miniBatchCompensator; k++)
						{
			
			
					
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				FORWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
							// TODO Auto-generated method stub
							//1. Sigmoid activation function = 1/(1+e^(-z))
							//		---e is the Eulers number = 2.71828
							//		---z is the DOT PRODUCT of the vector W with vector X plus the bias
							//		---vector W is a vector containing all the weights
							//		---vector X is a vector containing all the inputs
							//		---bias is the assigned bias
							//
							//		To do this we need to first have a way to set the weights and inputs
							//		
							//		Then we can feed those vectors into a method that finds the Z
							//
							//		Then we can insert Z into the SAF in order to spit out the activation value
							// 		
							//		Then we run the results back through
							
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							//System.out.println(zMiniBatchCompensator);
							//sets the first input set and uses it below
							
							currentHiddenLayerInputSet =  hiddenLayerInputVector[k].clone();
							
						
							//System.out.println("I am within the INNER loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//****CURRENT ERROR IS HERE SOMETHING GOES OUT OF BOUNDS OF 10 *********
							//currentHiddenLayerZSets uses the zMiniBatchCompensator because k is situated in a way to go 0 1 2 3... while the max zsets we can hold are only 0 1
							currentHiddenLayerZSets[zMiniBatchCompensator] = zAssignment(hiddenLayerWeightVector, currentHiddenLayerInputSet, hiddenLayerBiasVector);
							
							//Assigns the outcome of throwing the currentHiddenLayerZSets through the sigmoidal function
							currentHiddenLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentHiddenLayerZSets[zMiniBatchCompensator]);
					
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Grabs the current input set we acquired from the activation sums of the previous layer
							currentOutputLayerInputSet = currentHiddenLayerActivationSumSets[zMiniBatchCompensator].clone();
							
							//Throws the new input set through the zAssignment function and grabs the output layer Z sets from the equation
							currentOutputLayerZSets[zMiniBatchCompensator] = zAssignment(outputLayerWeightVector, currentOutputLayerInputSet, outputLayerBiasVector);
												
							//Temporarily stores the output layer activation sums after going through the sigmoidal function
							currentOutputLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentOutputLayerZSets[zMiniBatchCompensator]);
			
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				BACKWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							
							//Next is to create the back propagation things that HAVE SEPERATE EQUATIONS FOR INTENRAL AND EXTERNAL LAYERS !!!!!
							//The equation for the error/ THE BIASGRADIENT in the output layer is L = (a-y) * a * (1-a) 
							// a being the outputLayerActivationSum
							//y being the desired activation value
							
							//Method that grabs the current desired input sets to use later within the loop
							currentOutputLayerTrainingSets[zMiniBatchCompensator] = outputLayerTrainingVector[k].clone();
							
							//SO THE TABLE SHOWS A CONSISTENT PATTERN WHICH ISNT GOOD IT SHOULD BE COMPLETELY RANDOM...
							//Increments the corresponding numbers aka if this is the correct output and the total
							
							//Max element in the list
							//System.out.println("Zmini is " + zMiniBatchCompensator);
							maxElementIndex = maxElementInArray(currentOutputLayerActivationSumSets[zMiniBatchCompensator]);
							desiredIsZero = 1;
							
							//If its desired is 1 then increment total, then if guess is max of the set then also increment it
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][1] == 1.0)
							{
								totalOnes = 1 + totalOnes;
								if (maxElementIndex == 1)
								{
									totalCorrectlyGuessedOnes = 1 + totalCorrectlyGuessedOnes;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][2] == 2.0)
							{
								totalTwos = 1 + totalTwos;
								if(maxElementIndex == 2)
								{
									totalCorrectlyGuessedTwos = 1 + totalCorrectlyGuessedTwos;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][3] == 3.0)
							{
								totalThrees = 1 + totalThrees;
								if(maxElementIndex == 3)
								{
									totalCorrectlyGuessedThrees = 1 + totalCorrectlyGuessedThrees;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][4] == 4.0)
							{
								totalFours = 1 + totalFours;
								if(maxElementIndex == 4)
								{
									totalCorrectlyGuessedFours = 1 + totalCorrectlyGuessedFours;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][5] == 5.0)
							{
								totalFives = 1 + totalFives;
								if(maxElementIndex == 5)
								{
									totalCorrectlyGuessedFives = 1 + totalCorrectlyGuessedFives;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][6] == 6.0)
							{
								totalSixs = 1 + totalSixs;
								if(maxElementIndex == 6)
								{
									totalCorrectlyGuessedSixs = 1 + totalCorrectlyGuessedSixs;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][7] == 7.0)
							{
								totalSevens = 1 + totalSevens;
								if(maxElementIndex == 7)
								{
									totalCorrectlyGuessedSevens = 1 + totalCorrectlyGuessedSevens;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][8] == 8.0)
							{
								totalEights = 1 + totalEights;
								if(maxElementIndex == 8)
								{
									totalCorrectlyGuessedEights = 1 + totalCorrectlyGuessedEights;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][9] == 9.0)
							{
								totalNines = 1 + totalNines;
								if(maxElementIndex == 9)
								{
									totalCorrectlyGuessedNines= 1 + totalCorrectlyGuessedNines;
								}
								desiredIsZero = 0;
							}
							
							if(desiredIsZero == 1)
							{
								totalZeros = 1 + totalZeros;
								if(maxElementIndex == 0)
								{
									totalCorrectlyGuessedZeros = 1 + totalCorrectlyGuessedZeros;
								}
							}
							
							maxElementIndex = -1;
									
						
							//Method that temporarily grabs the bias gradients for later use in the loops
							currentOutputLayerBiasGradientSets[zMiniBatchCompensator] = outputLayerBiasGradientFunction(currentOutputLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerTrainingSets[zMiniBatchCompensator]);
							
							//After the bias gradient we would need to then find the outputWeightGradient which is the previous's layer activation sum * the gradient bias
							currentOutputLayerWeightGradientSets[zMiniBatchCompensator] = outputLayerWeightGradientFunction(currentOutputLayerBiasGradientSets[zMiniBatchCompensator], currentHiddenLayerActivationSumSets[zMiniBatchCompensator]);
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Temporarily stores the current hidden layer bias gradient for later use in revising the biases
							currentHiddenLayerBiasGradientSets[zMiniBatchCompensator] = hiddenLayerBiasGradientFunction( outputLayerWeightVector, currentHiddenLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//Gathers the hidden layers weight gradient to use in the future		
							currentHiddenLayerWeightGradientSets[zMiniBatchCompensator] = hiddenLayerWeightGradientFunction(currentHiddenLayerInputSet, currentHiddenLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//used to alternate between the 10 training sets that we use
							zMiniBatchCompensator = 1 + zMiniBatchCompensator;
						}
						zMiniBatchCompensator = 0;
						
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						//REVISING BIASES AND WEIGHTS          						    															  //
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						
						//Revised weights for the hidden layer TILES BF26 TO BI30
						revisedHiddenLayerWeightVector = RevisedWeightVectorFunction(currentHiddenLayerWeightGradientSets, hiddenLayerWeightVector);
						
						//Revised weights for the output layer TILES BF35 TO BH36
						revisedOutputLayerWeightVector = RevisedWeightVectorFunction(currentOutputLayerWeightGradientSets, outputLayerWeightVector);
						
						//Revised biases for the hidden layer tiles BD27 to BD29
						revisedHiddenLayerBiasVector = RevisedBiasVectorFunction(currentHiddenLayerBiasGradientSets, hiddenLayerBiasVector); 
						
						//Revised biases for the output layer tiles BD35 to BD36
						//double[] outputLayerRevisedBiasVector = RevisedBiasVectorFunction(outputLayerBiasGradientTrainingCase2, outputLayerBiasGradient, outputLayerBiasVector);
						revisedOutputLayerBiasVector = RevisedBiasVectorFunction(currentOutputLayerBiasGradientSets, outputLayerBiasVector);
						
						
						
						
			
						
						
						
						
						
						
						
						
						
						
						//SETTING THE NEW WEIGHTS AND BIASES
						
						//setting new revised hidden layer weights
						hiddenLayerWeightVector = revisedHiddenLayerWeightVector.clone();
						
						//setting new revised output layer weights
						outputLayerWeightVector = revisedOutputLayerWeightVector.clone();
						
						//setting new revised hidden layer biases
						hiddenLayerBiasVector = revisedHiddenLayerBiasVector.clone();
						
						//setting new revised output layer biases
						outputLayerBiasVector = revisedOutputLayerBiasVector.clone();
						
						//Used to alternate between each training set if theres 2
						miniBatchCompensator = numOfTrainingSetsPerMinibatch + miniBatchCompensator;
						//System.out.println("REVISEDminiBatchCompensator = " + miniBatchCompensator);
					}
					
					//Storing the final weights and biases to be used later
					recordedHiddenLayerWeightVector = hiddenLayerWeightVector.clone();
					recordedOutputLayerWeightVector = outputLayerWeightVector.clone();
					recordedHiddenLayerBiasVector = hiddenLayerBiasVector.clone();
					recordedOutputLayerBiasVector = outputLayerBiasVector.clone();
					
					totalCorrectlyGuessedDigits = totalCorrectlyGuessedZeros+totalCorrectlyGuessedOnes+totalCorrectlyGuessedTwos+totalCorrectlyGuessedThrees+totalCorrectlyGuessedFours+totalCorrectlyGuessedFives+totalCorrectlyGuessedSixs+totalCorrectlyGuessedSevens+totalCorrectlyGuessedEights+totalCorrectlyGuessedNines;
					totalDigits = totalZeros + totalOnes + totalTwos + totalThrees + totalFours + totalFives + totalSixs + totalSevens + totalEights + totalNines;
					Accuracy = (Double.valueOf(totalCorrectlyGuessedDigits)/Double.valueOf(totalDigits));
					//OUTPUT:
					System.out.println("Epoch: " + i);
					System.out.println("	0: "+ totalCorrectlyGuessedZeros+"/"+ totalZeros);
					System.out.println("	1: " + totalCorrectlyGuessedOnes +"/"+ totalOnes);
					System.out.println("	2: "+ totalCorrectlyGuessedTwos +"/"+ totalTwos);
					System.out.println("	3: "+ totalCorrectlyGuessedThrees +"/"+ totalThrees);
					System.out.println("	4: "+ totalCorrectlyGuessedFours +"/"+ totalFours);
					System.out.println("	5: "+ totalCorrectlyGuessedFives +"/"+ totalFives);
					System.out.println("	6: "+ totalCorrectlyGuessedSixs +"/"+ totalSixs);
					System.out.println("	7: "+ totalCorrectlyGuessedSevens +"/"+ totalSevens);
					System.out.println("	8: "+ totalCorrectlyGuessedEights +"/"+ totalEights);
					System.out.println("	9: "+ totalCorrectlyGuessedNines +"/"+ totalNines);
					System.out.println("	Accuracy = "+ totalCorrectlyGuessedDigits +"/" + totalDigits + " = " + Accuracy*100 + "%");
					
				}
				
				isMode7Availiable = 1;
			}
			}
if(mode == 4)
{
	if(isModeAvailiable == 0)
	{
		System.out.println("You have not selected any weights yet...");
	}
	else
	{
				
			
				numOfEpochs = 1;
				numOfMinibatches = 1000;
				//setting new revised hidden layer weights
				//Converts CSV To an dimensional Array (NOTE): The reason the dimension adds a 1 is because the label is at the front of the input set
				double[][]  convertedInputSets = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer + 1];
				convertedInputSets = convertCsvToArray(testingInputFilePath, numOfMinibatches, numOfTrainingSetsPerMinibatch, numOfNodesInInputLayer);
				
				//Randomly shuffled inputsets
				convertedInputSets = shuffleArray(convertedInputSets);
				
				

				
				//This is also how many training sets we want per minibatch
				double[][]  hiddenLayerInputVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer];
				hiddenLayerInputVector = inputVectorGrabber(convertedInputSets);
				
				//array of the hidden layers initial weight vector
				double[][]  hiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				hiddenLayerWeightVector = recordedHiddenLayerWeightVector.clone();
				
				//Empty revisingWeightVector that will be used to store the revised weights before cloning them to the original array to replace the old weights
				double[][] revisedHiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				
				//Array that is the initial biases manually set up using our test inputs... will be changed to random later
				double[] hiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				hiddenLayerBiasVector = hiddenLayerBiasVectorGrabber(numOfNodesInHiddenLayer);
				
				//Empty array used to store the revised hidden layer bias vector for later use
				double[] revisedHiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				
				//Array of the output layers initial weight vector
				double[][] outputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
				outputLayerWeightVector = recordedOutputLayerWeightVector.clone();
				
				//Empty array that will be used to store the revised output layer weight vector
				double[][] revisedOutputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
			
				//Array of the output layers initial biases
				double[] outputLayerBiasVector = new double[numOfNodesInOutputLayer];
				outputLayerBiasVector = outputLayerBiasVectorGrabber(numOfNodesInOutputLayer);
				
				double[] revisedOutputLayerBiasVector = new double[numOfNodesInOutputLayer];
				
				//Array of the output layers desired outputs for training
				double[][] outputLayerTrainingVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				outputLayerTrainingVector = outputLayerTrainingVectorGrabber(numOfNodesInOutputLayer, numOfMinibatches,numOfTrainingSetsPerMinibatch, convertedInputSets);
			
				
				//Compensator for cases in which we have multiple training sets per minibatch
				int miniBatchCompensator = 0;
				
				int zMiniBatchCompensator = 0;
				
				//Array containing the current training sets
				double[][] currentHiddenLayerZSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInHiddenLayer];
				
				//This only intakes 10 of the 60000 sets we have so we situate this to randomly choose 10 input sets
				//Initializes an array we will use as our current input set
				double[] currentHiddenLayerInputSet = new double[numOfNodesInInputLayer];
				
				//This is used to temporarily store the hidden layer activation sums
				double[][] currentHiddenLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//This is used to temporarily store the current output layer Z
				double[][] currentOutputLayerZSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Initializes an array we will use as our current input set
				double[] currentOutputLayerInputSet = new double[numOfNodesInOutputLayer];
			
				//This is used to temporarily store the output layer activation sums
				double[][] currentOutputLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Temporarily grabs the currently used desired output from the premade array
				double[][] currentOutputLayerTrainingSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//Temporarily stores the output of the bias gradients for that pass to use later in the loop
				double[][] currentOutputLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//DIMENSIONS
				//Temporarily stores the output of the weight gradient once everything has been thrown through the equations
				double[][][] currentOutputLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer][numOfNodesInHiddenLayer];	
			
				//Temporarily stores the output of the hidden Layer bias gradient functions for use in the loop
				double[][] currentHiddenLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//Temporarily stores the output of the hidden Layer weight gradient for use in the loop and revision
				double[][][] currentHiddenLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length][hiddenLayerInputVector.length];
				
				//System.out.println(hiddenLayerInputVector[6000]);
				
				
				
				
				
				
				
				
				
				
				
				
				//THIS IS FOR FUTURE AUSTIN... THERE IS AN ISSUE... WE NEED THE LOOPS TO DO 10 TRAINING SETS THEN CHANGE MINIBATCHES... SO 0,1,2,3,4,5,6,7,8,9... 0,1,2,3... etc etc
				//The issue is that the zCompensator is running after only 2 training sets which is wrong and throws it to become 10 it needs to only change to 10 ONCE we looped through
				//all the initial training sets first so I DO 10 SETS then say ok make my zCompensator 10 etc etc
				//
				
				
				
				
				
				
				
				
				
				
				
				
				
				//Counts total number of digits
				int totalZeros = 0;
				int totalOnes = 0;
				int totalTwos = 0;
				int totalThrees = 0;
				int totalFours = 0;
				int totalFives = 0;
				int totalSixs = 0;
				int totalSevens = 0;
				int totalEights = 0;
				int totalNines = 0;
				
				//Counts total number of correct digits
				int totalCorrectlyGuessedZeros = 0;
				int totalCorrectlyGuessedOnes = 0;
				int totalCorrectlyGuessedTwos = 0;
				int totalCorrectlyGuessedThrees = 0;
				int totalCorrectlyGuessedFours = 0;
				int totalCorrectlyGuessedFives = 0;
				int totalCorrectlyGuessedSixs = 0;
				int totalCorrectlyGuessedSevens = 0;
				int totalCorrectlyGuessedEights = 0;
				int totalCorrectlyGuessedNines = 0;
				
				int totalCorrectlyGuessedDigits = 0;
				double Accuracy = 0;
				int totalDigits = 0;
				
				//Checks if 0 is the current number
				int desiredIsZero = 0;
				
				double maxElementIndex = -1;
				
			
				//BIG BOY LOOP THAT DOES IT ALL
				//LOOP FOR EPOCHS 1
				for (int i = 0; i < numOfEpochs; i++)
				{
			
					//RESETTING THE COUNTS
					 totalZeros = 0;
					 totalOnes = 0;
					 totalTwos = 0;
					 totalThrees = 0;
					 totalFours = 0;
					 totalFives = 0;
					 totalSixs = 0;
					 totalSevens = 0;
					 totalEights = 0;
					 totalNines = 0;
					
					//RESETTING THE COUNTS
					 totalCorrectlyGuessedZeros = 0;
					 totalCorrectlyGuessedOnes = 0;
					 totalCorrectlyGuessedTwos = 0;
					 totalCorrectlyGuessedThrees = 0;
					 totalCorrectlyGuessedFours = 0;
					 totalCorrectlyGuessedFives = 0;
					 totalCorrectlyGuessedSixs = 0;
					 totalCorrectlyGuessedSevens = 0;
					 totalCorrectlyGuessedEights = 0;
					 totalCorrectlyGuessedNines = 0;
					 
					 totalCorrectlyGuessedDigits = 0;
					 Accuracy = 0;
					 totalDigits = 0;
					 
					 miniBatchCompensator = 0;
					 maxElementIndex = -1;
					 
					//loop for each minibatch
					for (int j = 0; j < numOfMinibatches; j++)
					{
						//System.out.println("I am within the outer loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//loop for each training loop
						//System.out.println("K should have gone to 10");
						//System.out.println("Comp is : " + miniBatchCompensator);
						
						
						//ERROR: THIS LOOP IS GOING 6000 TIMES... which makes sense it should but the loop below isnt incrementing that at all.... 
						//			essientially its only counting 10 digits then saying eh who cares... FIX THE MINI BATCH COMPENSATOR IS 0 INITIALLY THIS IS WHY
						
						
						for (int k = miniBatchCompensator; k < numOfTrainingSetsPerMinibatch + miniBatchCompensator; k++)
						{
			
			
					
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				FORWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
							// TODO Auto-generated method stub
							//1. Sigmoid activation function = 1/(1+e^(-z))
							//		---e is the Eulers number = 2.71828
							//		---z is the DOT PRODUCT of the vector W with vector X plus the bias
							//		---vector W is a vector containing all the weights
							//		---vector X is a vector containing all the inputs
							//		---bias is the assigned bias
							//
							//		To do this we need to first have a way to set the weights and inputs
							//		
							//		Then we can feed those vectors into a method that finds the Z
							//
							//		Then we can insert Z into the SAF in order to spit out the activation value
							// 		
							//		Then we run the results back through
							
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							//System.out.println(zMiniBatchCompensator);
							//sets the first input set and uses it below
							
							currentHiddenLayerInputSet =  hiddenLayerInputVector[k].clone();
							
						
							//System.out.println("I am within the INNER loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//****CURRENT ERROR IS HERE SOMETHING GOES OUT OF BOUNDS OF 10 *********
							//currentHiddenLayerZSets uses the zMiniBatchCompensator because k is situated in a way to go 0 1 2 3... while the max zsets we can hold are only 0 1
							currentHiddenLayerZSets[zMiniBatchCompensator] = zAssignment(hiddenLayerWeightVector, currentHiddenLayerInputSet, hiddenLayerBiasVector);
							
							//Assigns the outcome of throwing the currentHiddenLayerZSets through the sigmoidal function
							currentHiddenLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentHiddenLayerZSets[zMiniBatchCompensator]);
					
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Grabs the current input set we acquired from the activation sums of the previous layer
							currentOutputLayerInputSet = currentHiddenLayerActivationSumSets[zMiniBatchCompensator].clone();
							
							//Throws the new input set through the zAssignment function and grabs the output layer Z sets from the equation
							currentOutputLayerZSets[zMiniBatchCompensator] = zAssignment(outputLayerWeightVector, currentOutputLayerInputSet, outputLayerBiasVector);
												
							//Temporarily stores the output layer activation sums after going through the sigmoidal function
							currentOutputLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentOutputLayerZSets[zMiniBatchCompensator]);
			
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				BACKWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							
							//Next is to create the back propagation things that HAVE SEPERATE EQUATIONS FOR INTENRAL AND EXTERNAL LAYERS !!!!!
							//The equation for the error/ THE BIASGRADIENT in the output layer is L = (a-y) * a * (1-a) 
							// a being the outputLayerActivationSum
							//y being the desired activation value
							
							//Method that grabs the current desired input sets to use later within the loop
							currentOutputLayerTrainingSets[zMiniBatchCompensator] = outputLayerTrainingVector[k].clone();
							
							//SO THE TABLE SHOWS A CONSISTENT PATTERN WHICH ISNT GOOD IT SHOULD BE COMPLETELY RANDOM...
							//Increments the corresponding numbers aka if this is the correct output and the total
							
							//Max element in the list
							//System.out.println("Zmini is " + zMiniBatchCompensator);
							maxElementIndex = maxElementInArray(currentOutputLayerActivationSumSets[zMiniBatchCompensator]);
							desiredIsZero = 1;
							
							//If its desired is 1 then increment total, then if guess is max of the set then also increment it
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][1] == 1.0)
							{
								totalOnes = 1 + totalOnes;
								if (maxElementIndex == 1)
								{
									totalCorrectlyGuessedOnes = 1 + totalCorrectlyGuessedOnes;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][2] == 2.0)
							{
								totalTwos = 1 + totalTwos;
								if(maxElementIndex == 2)
								{
									totalCorrectlyGuessedTwos = 1 + totalCorrectlyGuessedTwos;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][3] == 3.0)
							{
								totalThrees = 1 + totalThrees;
								if(maxElementIndex == 3)
								{
									totalCorrectlyGuessedThrees = 1 + totalCorrectlyGuessedThrees;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][4] == 4.0)
							{
								totalFours = 1 + totalFours;
								if(maxElementIndex == 4)
								{
									totalCorrectlyGuessedFours = 1 + totalCorrectlyGuessedFours;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][5] == 5.0)
							{
								totalFives = 1 + totalFives;
								if(maxElementIndex == 5)
								{
									totalCorrectlyGuessedFives = 1 + totalCorrectlyGuessedFives;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][6] == 6.0)
							{
								totalSixs = 1 + totalSixs;
								if(maxElementIndex == 6)
								{
									totalCorrectlyGuessedSixs = 1 + totalCorrectlyGuessedSixs;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][7] == 7.0)
							{
								totalSevens = 1 + totalSevens;
								if(maxElementIndex == 7)
								{
									totalCorrectlyGuessedSevens = 1 + totalCorrectlyGuessedSevens;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][8] == 8.0)
							{
								totalEights = 1 + totalEights;
								if(maxElementIndex == 8)
								{
									totalCorrectlyGuessedEights = 1 + totalCorrectlyGuessedEights;
								}
								desiredIsZero = 0;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][9] == 9.0)
							{
								totalNines = 1 + totalNines;
								if(maxElementIndex == 9)
								{
									totalCorrectlyGuessedNines= 1 + totalCorrectlyGuessedNines;
								}
								desiredIsZero = 0;
							}
							
							if(desiredIsZero == 1)
							{
								totalZeros = 1 + totalZeros;
								if(maxElementIndex == 0)
								{
									totalCorrectlyGuessedZeros = 1 + totalCorrectlyGuessedZeros;
								}
							}
							
							maxElementIndex = -1;
									
						
							//Method that temporarily grabs the bias gradients for later use in the loops
							currentOutputLayerBiasGradientSets[zMiniBatchCompensator] = outputLayerBiasGradientFunction(currentOutputLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerTrainingSets[zMiniBatchCompensator]);
							
							//After the bias gradient we would need to then find the outputWeightGradient which is the previous's layer activation sum * the gradient bias
							currentOutputLayerWeightGradientSets[zMiniBatchCompensator] = outputLayerWeightGradientFunction(currentOutputLayerBiasGradientSets[zMiniBatchCompensator], currentHiddenLayerActivationSumSets[zMiniBatchCompensator]);
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Temporarily stores the current hidden layer bias gradient for later use in revising the biases
							currentHiddenLayerBiasGradientSets[zMiniBatchCompensator] = hiddenLayerBiasGradientFunction( outputLayerWeightVector, currentHiddenLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//Gathers the hidden layers weight gradient to use in the future		
							currentHiddenLayerWeightGradientSets[zMiniBatchCompensator] = hiddenLayerWeightGradientFunction(currentHiddenLayerInputSet, currentHiddenLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//used to alternate between the 10 training sets that we use
							zMiniBatchCompensator = 1 + zMiniBatchCompensator;
						}
						zMiniBatchCompensator = 0;
						
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						//REVISING BIASES AND WEIGHTS          						    															  //
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						
						//Revised weights for the hidden layer TILES BF26 TO BI30
						revisedHiddenLayerWeightVector = RevisedWeightVectorFunction(currentHiddenLayerWeightGradientSets, hiddenLayerWeightVector);
						
						//Revised weights for the output layer TILES BF35 TO BH36
						revisedOutputLayerWeightVector = RevisedWeightVectorFunction(currentOutputLayerWeightGradientSets, outputLayerWeightVector);
						
						//Revised biases for the hidden layer tiles BD27 to BD29
						revisedHiddenLayerBiasVector = RevisedBiasVectorFunction(currentHiddenLayerBiasGradientSets, hiddenLayerBiasVector); 
						
						//Revised biases for the output layer tiles BD35 to BD36
						//double[] outputLayerRevisedBiasVector = RevisedBiasVectorFunction(outputLayerBiasGradientTrainingCase2, outputLayerBiasGradient, outputLayerBiasVector);
						revisedOutputLayerBiasVector = RevisedBiasVectorFunction(currentOutputLayerBiasGradientSets, outputLayerBiasVector);
						
						
						
						
			
						
						
						
						
						
						
						
						
						
						
						//SETTING THE NEW WEIGHTS AND BIASES
						
						//setting new revised hidden layer weights
						hiddenLayerWeightVector = revisedHiddenLayerWeightVector.clone();
						
						//setting new revised output layer weights
						outputLayerWeightVector = revisedOutputLayerWeightVector.clone();
						
						//setting new revised hidden layer biases
						hiddenLayerBiasVector = revisedHiddenLayerBiasVector.clone();
						
						//setting new revised output layer biases
						outputLayerBiasVector = revisedOutputLayerBiasVector.clone();
						
						//Used to alternate between each training set if theres 2
						miniBatchCompensator = numOfTrainingSetsPerMinibatch + miniBatchCompensator;
						//System.out.println("REVISEDminiBatchCompensator = " + miniBatchCompensator);
					}
					
					//Storing the final weights and biases to be used later
					recordedHiddenLayerWeightVector = hiddenLayerWeightVector.clone();
					recordedOutputLayerWeightVector = outputLayerWeightVector.clone();
					recordedHiddenLayerBiasVector = hiddenLayerBiasVector.clone();
					recordedOutputLayerBiasVector = outputLayerBiasVector.clone();
					
					totalCorrectlyGuessedDigits = totalCorrectlyGuessedZeros+totalCorrectlyGuessedOnes+totalCorrectlyGuessedTwos+totalCorrectlyGuessedThrees+totalCorrectlyGuessedFours+totalCorrectlyGuessedFives+totalCorrectlyGuessedSixs+totalCorrectlyGuessedSevens+totalCorrectlyGuessedEights+totalCorrectlyGuessedNines;
					totalDigits = totalZeros + totalOnes + totalTwos + totalThrees + totalFours + totalFives + totalSixs + totalSevens + totalEights + totalNines;
					Accuracy = (Double.valueOf(totalCorrectlyGuessedDigits)/Double.valueOf(totalDigits));
					//OUTPUT:
					System.out.println("Epoch: " + i);
					System.out.println("	0: "+ totalCorrectlyGuessedZeros+"/"+ totalZeros);
					System.out.println("	1: " + totalCorrectlyGuessedOnes +"/"+ totalOnes);
					System.out.println("	2: "+ totalCorrectlyGuessedTwos +"/"+ totalTwos);
					System.out.println("	3: "+ totalCorrectlyGuessedThrees +"/"+ totalThrees);
					System.out.println("	4: "+ totalCorrectlyGuessedFours +"/"+ totalFours);
					System.out.println("	5: "+ totalCorrectlyGuessedFives +"/"+ totalFives);
					System.out.println("	6: "+ totalCorrectlyGuessedSixs +"/"+ totalSixs);
					System.out.println("	7: "+ totalCorrectlyGuessedSevens +"/"+ totalSevens);
					System.out.println("	8: "+ totalCorrectlyGuessedEights +"/"+ totalEights);
					System.out.println("	9: "+ totalCorrectlyGuessedNines +"/"+ totalNines);
					System.out.println("	Accuracy = "+ totalCorrectlyGuessedDigits +"/" + totalDigits + " = " + Accuracy*100 + "%");
					
				}
				
				isMode7Availiable = 1;
	}
						}
if(mode == 5)
{
	if(isModeAvailiable == 0)
		{
			System.out.println("You have not selected any weights yet...");
		}
		else
		{
				
				numOfEpochs = 1;
				numOfMinibatches = 1000;
				//setting new revised hidden layer weights
				//Converts CSV To an dimensional Array (NOTE): The reason the dimension adds a 1 is because the label is at the front of the input set
				double[][]  convertedInputSets = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer + 1];
				convertedInputSets = convertCsvToArray(testingInputFilePath, numOfMinibatches, numOfTrainingSetsPerMinibatch, numOfNodesInInputLayer);
				
				//Randomly shuffled inputsets
				convertedInputSets = shuffleArray(convertedInputSets);
				
				
				 
					
				
				//This is also how many training sets we want per minibatch
				double[][]  hiddenLayerInputVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer];
				hiddenLayerInputVector = inputVectorGrabber(convertedInputSets);
				
				//array of the hidden layers initial weight vector
				double[][]  hiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				hiddenLayerWeightVector = recordedHiddenLayerWeightVector.clone();
				
				//Empty revisingWeightVector that will be used to store the revised weights before cloning them to the original array to replace the old weights
				double[][] revisedHiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				
				//Array that is the initial biases manually set up using our test inputs... will be changed to random later
				double[] hiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				hiddenLayerBiasVector = hiddenLayerBiasVectorGrabber(numOfNodesInHiddenLayer);
				
				//Empty array used to store the revised hidden layer bias vector for later use
				double[] revisedHiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				
				//Array of the output layers initial weight vector
				double[][] outputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
				outputLayerWeightVector = recordedOutputLayerWeightVector.clone();
				
				//Empty array that will be used to store the revised output layer weight vector
				double[][] revisedOutputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
			
				//Array of the output layers initial biases
				double[] outputLayerBiasVector = new double[numOfNodesInOutputLayer];
				outputLayerBiasVector = outputLayerBiasVectorGrabber(numOfNodesInOutputLayer);
				
				double[] revisedOutputLayerBiasVector = new double[numOfNodesInOutputLayer];
				
				//Array of the output layers desired outputs for training
				double[][] outputLayerTrainingVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				outputLayerTrainingVector = outputLayerTrainingVectorGrabber(numOfNodesInOutputLayer, numOfMinibatches,numOfTrainingSetsPerMinibatch, convertedInputSets);
			
				
				//Compensator for cases in which we have multiple training sets per minibatch
				int miniBatchCompensator = 0;
				
				int zMiniBatchCompensator = 0;
				
				//Array containing the current training sets
				double[][] currentHiddenLayerZSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInHiddenLayer];
				
				//This only intakes 10 of the 60000 sets we have so we situate this to randomly choose 10 input sets
				//Initializes an array we will use as our current input set
				double[] currentHiddenLayerInputSet = new double[numOfNodesInInputLayer];
				
				//This is used to temporarily store the hidden layer activation sums
				double[][] currentHiddenLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//This is used to temporarily store the current output layer Z
				double[][] currentOutputLayerZSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Initializes an array we will use as our current input set
				double[] currentOutputLayerInputSet = new double[numOfNodesInOutputLayer];
			
				//This is used to temporarily store the output layer activation sums
				double[][] currentOutputLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Temporarily grabs the currently used desired output from the premade array
				double[][] currentOutputLayerTrainingSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//Temporarily stores the output of the bias gradients for that pass to use later in the loop
				double[][] currentOutputLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//DIMENSIONS
				//Temporarily stores the output of the weight gradient once everything has been thrown through the equations
				double[][][] currentOutputLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer][numOfNodesInHiddenLayer];	
			
				//Temporarily stores the output of the hidden Layer bias gradient functions for use in the loop
				double[][] currentHiddenLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//Temporarily stores the output of the hidden Layer weight gradient for use in the loop and revision
				double[][][] currentHiddenLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length][hiddenLayerInputVector.length];
				
				//System.out.println(hiddenLayerInputVector[6000]);
				
				
				
				
				
				
				
				
				
				
				
				
				//THIS IS FOR FUTURE AUSTIN... THERE IS AN ISSUE... WE NEED THE LOOPS TO DO 10 TRAINING SETS THEN CHANGE MINIBATCHES... SO 0,1,2,3,4,5,6,7,8,9... 0,1,2,3... etc etc
				//The issue is that the zCompensator is running after only 2 training sets which is wrong and throws it to become 10 it needs to only change to 10 ONCE we looped through
				//all the initial training sets first so I DO 10 SETS then say ok make my zCompensator 10 etc etc
				//
				
				
				
				
				
				
				
				
				
				
				
				
				
				//Counts total number of digits
				int totalZeros = 0;
				int totalOnes = 0;
				int totalTwos = 0;
				int totalThrees = 0;
				int totalFours = 0;
				int totalFives = 0;
				int totalSixs = 0;
				int totalSevens = 0;
				int totalEights = 0;
				int totalNines = 0;
				
				//Counts total number of correct digits
				int totalCorrectlyGuessedZeros = 0;
				int totalCorrectlyGuessedOnes = 0;
				int totalCorrectlyGuessedTwos = 0;
				int totalCorrectlyGuessedThrees = 0;
				int totalCorrectlyGuessedFours = 0;
				int totalCorrectlyGuessedFives = 0;
				int totalCorrectlyGuessedSixs = 0;
				int totalCorrectlyGuessedSevens = 0;
				int totalCorrectlyGuessedEights = 0;
				int totalCorrectlyGuessedNines = 0;
				
				int totalCorrectlyGuessedDigits = 0;
				double Accuracy = 0;
				int totalDigits = 0;
				
				//Checks if 0 is the current number
				int desiredIsZero = 0;
				int desiredIs = 0;
				
				double maxElementIndex = -1;
				
			
				//BIG BOY LOOP THAT DOES IT ALL
				//LOOP FOR EPOCHS 1
				for (int i = 0; i < numOfEpochs; i++)
				{
			
					//RESETTING THE COUNTS
					 totalZeros = 0;
					 totalOnes = 0;
					 totalTwos = 0;
					 totalThrees = 0;
					 totalFours = 0;
					 totalFives = 0;
					 totalSixs = 0;
					 totalSevens = 0;
					 totalEights = 0;
					 totalNines = 0;
					
					//RESETTING THE COUNTS
					 totalCorrectlyGuessedZeros = 0;
					 totalCorrectlyGuessedOnes = 0;
					 totalCorrectlyGuessedTwos = 0;
					 totalCorrectlyGuessedThrees = 0;
					 totalCorrectlyGuessedFours = 0;
					 totalCorrectlyGuessedFives = 0;
					 totalCorrectlyGuessedSixs = 0;
					 totalCorrectlyGuessedSevens = 0;
					 totalCorrectlyGuessedEights = 0;
					 totalCorrectlyGuessedNines = 0;
					 
					 totalCorrectlyGuessedDigits = 0;
					 Accuracy = 0;
					 totalDigits = 0;
					 
					 miniBatchCompensator = 0;
					 maxElementIndex = -1;
					 
					//loop for each minibatch
					for (int j = 0; j < numOfMinibatches; j++)
					{
						//System.out.println("I am within the outer loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//loop for each training loop
						//System.out.println("K should have gone to 10");
						//System.out.println("Comp is : " + miniBatchCompensator);
						
						
						//ERROR: THIS LOOP IS GOING 6000 TIMES... which makes sense it should but the loop below isnt incrementing that at all.... 
						//			essientially its only counting 10 digits then saying eh who cares... FIX THE MINI BATCH COMPENSATOR IS 0 INITIALLY THIS IS WHY
						
						
						for (int k = miniBatchCompensator; k < numOfTrainingSetsPerMinibatch + miniBatchCompensator; k++)
						{
			
			
					
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				FORWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
							// TODO Auto-generated method stub
							//1. Sigmoid activation function = 1/(1+e^(-z))
							//		---e is the Eulers number = 2.71828
							//		---z is the DOT PRODUCT of the vector W with vector X plus the bias
							//		---vector W is a vector containing all the weights
							//		---vector X is a vector containing all the inputs
							//		---bias is the assigned bias
							//
							//		To do this we need to first have a way to set the weights and inputs
							//		
							//		Then we can feed those vectors into a method that finds the Z
							//
							//		Then we can insert Z into the SAF in order to spit out the activation value
							// 		
							//		Then we run the results back through
							
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							//System.out.println(zMiniBatchCompensator);
							//sets the first input set and uses it below
							
							currentHiddenLayerInputSet =  hiddenLayerInputVector[k].clone();
							
						
							//System.out.println("I am within the INNER loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//****CURRENT ERROR IS HERE SOMETHING GOES OUT OF BOUNDS OF 10 *********
							//currentHiddenLayerZSets uses the zMiniBatchCompensator because k is situated in a way to go 0 1 2 3... while the max zsets we can hold are only 0 1
							currentHiddenLayerZSets[zMiniBatchCompensator] = zAssignment(hiddenLayerWeightVector, currentHiddenLayerInputSet, hiddenLayerBiasVector);
							
							//Assigns the outcome of throwing the currentHiddenLayerZSets through the sigmoidal function
							currentHiddenLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentHiddenLayerZSets[zMiniBatchCompensator]);
					
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Grabs the current input set we acquired from the activation sums of the previous layer
							currentOutputLayerInputSet = currentHiddenLayerActivationSumSets[zMiniBatchCompensator].clone();
							
							//Throws the new input set through the zAssignment function and grabs the output layer Z sets from the equation
							currentOutputLayerZSets[zMiniBatchCompensator] = zAssignment(outputLayerWeightVector, currentOutputLayerInputSet, outputLayerBiasVector);
												
							//Temporarily stores the output layer activation sums after going through the sigmoidal function
							currentOutputLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentOutputLayerZSets[zMiniBatchCompensator]);
			
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				BACKWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							
							//Next is to create the back propagation things that HAVE SEPERATE EQUATIONS FOR INTENRAL AND EXTERNAL LAYERS !!!!!
							//The equation for the error/ THE BIASGRADIENT in the output layer is L = (a-y) * a * (1-a) 
							// a being the outputLayerActivationSum
							//y being the desired activation value
							
							//Method that grabs the current desired input sets to use later within the loop
							currentOutputLayerTrainingSets[zMiniBatchCompensator] = outputLayerTrainingVector[k].clone();
							
							//SO THE TABLE SHOWS A CONSISTENT PATTERN WHICH ISNT GOOD IT SHOULD BE COMPLETELY RANDOM...
							//Increments the corresponding numbers aka if this is the correct output and the total
							
							//Max element in the list
							//System.out.println("Zmini is " + zMiniBatchCompensator);
							maxElementIndex = maxElementInArray(currentOutputLayerActivationSumSets[zMiniBatchCompensator]);
							desiredIsZero = 1;
							desiredIs = -1;
							
							
							
							
							//If its desired is 1 then increment total, then if guess is max of the set then also increment it
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][1] == 1.0)
							{
								totalOnes = 1 + totalOnes;
								if (maxElementIndex == 1)
								{
									totalCorrectlyGuessedOnes = 1 + totalCorrectlyGuessedOnes;
								}
								desiredIsZero = 0;
								desiredIs = 1;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][2] == 2.0)
							{
								totalTwos = 1 + totalTwos;
								if(maxElementIndex == 2)
								{
									totalCorrectlyGuessedTwos = 1 + totalCorrectlyGuessedTwos;
								}
								desiredIsZero = 0;
								desiredIs = 2;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][3] == 3.0)
							{
								totalThrees = 1 + totalThrees;
								if(maxElementIndex == 3)
								{
									totalCorrectlyGuessedThrees = 1 + totalCorrectlyGuessedThrees;
								}
								desiredIsZero = 0;
								desiredIs = 3;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][4] == 4.0)
							{
								totalFours = 1 + totalFours;
								if(maxElementIndex == 4)
								{
									totalCorrectlyGuessedFours = 1 + totalCorrectlyGuessedFours;
								}
								desiredIsZero = 0;
								desiredIs = 4;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][5] == 5.0)
							{
								totalFives = 1 + totalFives;
								if(maxElementIndex == 5)
								{
									totalCorrectlyGuessedFives = 1 + totalCorrectlyGuessedFives;
								}
								desiredIsZero = 0;
								desiredIs = 5;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][6] == 6.0)
							{
								totalSixs = 1 + totalSixs;
								if(maxElementIndex == 6)
								{
									totalCorrectlyGuessedSixs = 1 + totalCorrectlyGuessedSixs;
								}
								desiredIsZero = 0;
								desiredIs = 6;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][7] == 7.0)
							{
								totalSevens = 1 + totalSevens;
								if(maxElementIndex == 7)
								{
									totalCorrectlyGuessedSevens = 1 + totalCorrectlyGuessedSevens;
								}
								desiredIsZero = 0;
								desiredIs = 7;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][8] == 8.0)
							{
								totalEights = 1 + totalEights;
								if(maxElementIndex == 8)
								{
									totalCorrectlyGuessedEights = 1 + totalCorrectlyGuessedEights;
								}
								desiredIsZero = 0;
								desiredIs = 8;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][9] == 9.0)
							{
								totalNines = 1 + totalNines;
								if(maxElementIndex == 9)
								{
									totalCorrectlyGuessedNines= 1 + totalCorrectlyGuessedNines;
								}
								desiredIsZero = 0;
								desiredIs = 9;
							}
							
							if(desiredIsZero == 1)
							{
								totalZeros = 1 + totalZeros;
								if(maxElementIndex == 0)
								{
									totalCorrectlyGuessedZeros = 1 + totalCorrectlyGuessedZeros;
								}
								desiredIs = 0;
							}
							
							pressEnterToContinue();
							//Visually shows the current image and the guessed and desired values
							printVisualExample(currentHiddenLayerInputSet,maxElementIndex, desiredIs);
							
							
							pressEnterToContinue();
							
							maxElementIndex = -1;
									
						
							//Method that temporarily grabs the bias gradients for later use in the loops
							currentOutputLayerBiasGradientSets[zMiniBatchCompensator] = outputLayerBiasGradientFunction(currentOutputLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerTrainingSets[zMiniBatchCompensator]);
							
							//After the bias gradient we would need to then find the outputWeightGradient which is the previous's layer activation sum * the gradient bias
							currentOutputLayerWeightGradientSets[zMiniBatchCompensator] = outputLayerWeightGradientFunction(currentOutputLayerBiasGradientSets[zMiniBatchCompensator], currentHiddenLayerActivationSumSets[zMiniBatchCompensator]);
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Temporarily stores the current hidden layer bias gradient for later use in revising the biases
							currentHiddenLayerBiasGradientSets[zMiniBatchCompensator] = hiddenLayerBiasGradientFunction( outputLayerWeightVector, currentHiddenLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//Gathers the hidden layers weight gradient to use in the future		
							currentHiddenLayerWeightGradientSets[zMiniBatchCompensator] = hiddenLayerWeightGradientFunction(currentHiddenLayerInputSet, currentHiddenLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//used to alternate between the 10 training sets that we use
							zMiniBatchCompensator = 1 + zMiniBatchCompensator;
						}
						zMiniBatchCompensator = 0;
						
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						//REVISING BIASES AND WEIGHTS          						    															  //
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						
						//Revised weights for the hidden layer TILES BF26 TO BI30
						revisedHiddenLayerWeightVector = RevisedWeightVectorFunction(currentHiddenLayerWeightGradientSets, hiddenLayerWeightVector);
						
						//Revised weights for the output layer TILES BF35 TO BH36
						revisedOutputLayerWeightVector = RevisedWeightVectorFunction(currentOutputLayerWeightGradientSets, outputLayerWeightVector);
						
						//Revised biases for the hidden layer tiles BD27 to BD29
						revisedHiddenLayerBiasVector = RevisedBiasVectorFunction(currentHiddenLayerBiasGradientSets, hiddenLayerBiasVector); 
						
						//Revised biases for the output layer tiles BD35 to BD36
						//double[] outputLayerRevisedBiasVector = RevisedBiasVectorFunction(outputLayerBiasGradientTrainingCase2, outputLayerBiasGradient, outputLayerBiasVector);
						revisedOutputLayerBiasVector = RevisedBiasVectorFunction(currentOutputLayerBiasGradientSets, outputLayerBiasVector);
						
						
						
						
			
						
						
						
						
						
						
						
						
						
						
						//SETTING THE NEW WEIGHTS AND BIASES
						
						//setting new revised hidden layer weights
						hiddenLayerWeightVector = revisedHiddenLayerWeightVector.clone();
						
						//setting new revised output layer weights
						outputLayerWeightVector = revisedOutputLayerWeightVector.clone();
						
						//setting new revised hidden layer biases
						hiddenLayerBiasVector = revisedHiddenLayerBiasVector.clone();
						
						//setting new revised output layer biases
						outputLayerBiasVector = revisedOutputLayerBiasVector.clone();
						
						//Used to alternate between each training set if theres 2
						miniBatchCompensator = numOfTrainingSetsPerMinibatch + miniBatchCompensator;
						//System.out.println("REVISEDminiBatchCompensator = " + miniBatchCompensator);
					}
					
					//Storing the final weights and biases to be used later
					recordedHiddenLayerWeightVector = hiddenLayerWeightVector.clone();
					recordedOutputLayerWeightVector = outputLayerWeightVector.clone();
					recordedHiddenLayerBiasVector = hiddenLayerBiasVector.clone();
					recordedOutputLayerBiasVector = outputLayerBiasVector.clone();
					
				}
				
				isMode7Availiable = 1;
		}
						}
if(mode == 6)
						{
				
				numOfEpochs = 1;
				numOfMinibatches = 1000;
				//setting new revised hidden layer weights
				//Converts CSV To an dimensional Array (NOTE): The reason the dimension adds a 1 is because the label is at the front of the input set
				double[][]  convertedInputSets = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer + 1];
				convertedInputSets = convertCsvToArray(testingInputFilePath, numOfMinibatches, numOfTrainingSetsPerMinibatch, numOfNodesInInputLayer);
				
				//Randomly shuffled inputsets
				convertedInputSets = shuffleArray(convertedInputSets);
				
				
				
					
				
				//This is also how many training sets we want per minibatch
				double[][]  hiddenLayerInputVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer];
				hiddenLayerInputVector = inputVectorGrabber(convertedInputSets);
				
				//array of the hidden layers initial weight vector
				double[][]  hiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				hiddenLayerWeightVector = recordedHiddenLayerWeightVector.clone();
				
				//Empty revisingWeightVector that will be used to store the revised weights before cloning them to the original array to replace the old weights
				double[][] revisedHiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
				
				//Array that is the initial biases manually set up using our test inputs... will be changed to random later
				double[] hiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				hiddenLayerBiasVector = hiddenLayerBiasVectorGrabber(numOfNodesInHiddenLayer);
				
				//Empty array used to store the revised hidden layer bias vector for later use
				double[] revisedHiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
				
				//Array of the output layers initial weight vector
				double[][] outputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
				outputLayerWeightVector = recordedOutputLayerWeightVector.clone();
				
				//Empty array that will be used to store the revised output layer weight vector
				double[][] revisedOutputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
			
				//Array of the output layers initial biases
				double[] outputLayerBiasVector = new double[numOfNodesInOutputLayer];
				outputLayerBiasVector = outputLayerBiasVectorGrabber(numOfNodesInOutputLayer);
				
				double[] revisedOutputLayerBiasVector = new double[numOfNodesInOutputLayer];
				
				//Array of the output layers desired outputs for training
				double[][] outputLayerTrainingVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				outputLayerTrainingVector = outputLayerTrainingVectorGrabber(numOfNodesInOutputLayer, numOfMinibatches,numOfTrainingSetsPerMinibatch, convertedInputSets);
			
				
				//Compensator for cases in which we have multiple training sets per minibatch
				int miniBatchCompensator = 0;
				
				int zMiniBatchCompensator = 0;
				
				//Array containing the current training sets
				double[][] currentHiddenLayerZSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInHiddenLayer];
				
				//This only intakes 10 of the 60000 sets we have so we situate this to randomly choose 10 input sets
				//Initializes an array we will use as our current input set
				double[] currentHiddenLayerInputSet = new double[numOfNodesInInputLayer];
				
				//This is used to temporarily store the hidden layer activation sums
				double[][] currentHiddenLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//This is used to temporarily store the current output layer Z
				double[][] currentOutputLayerZSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Initializes an array we will use as our current input set
				double[] currentOutputLayerInputSet = new double[numOfNodesInOutputLayer];
			
				//This is used to temporarily store the output layer activation sums
				double[][] currentOutputLayerActivationSumSets = new double[numOfTrainingSetsPerMinibatch][outputLayerBiasVector.length];
				
				//Temporarily grabs the currently used desired output from the premade array
				double[][] currentOutputLayerTrainingSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//Temporarily stores the output of the bias gradients for that pass to use later in the loop
				double[][] currentOutputLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];
				
				//DIMENSIONS
				//Temporarily stores the output of the weight gradient once everything has been thrown through the equations
				double[][][] currentOutputLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer][numOfNodesInHiddenLayer];	
			
				//Temporarily stores the output of the hidden Layer bias gradient functions for use in the loop
				double[][] currentHiddenLayerBiasGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length];
				
				//Temporarily stores the output of the hidden Layer weight gradient for use in the loop and revision
				double[][][] currentHiddenLayerWeightGradientSets = new double[numOfTrainingSetsPerMinibatch][hiddenLayerBiasVector.length][hiddenLayerInputVector.length];
				
				//System.out.println(hiddenLayerInputVector[6000]);
				
				
				
				
				
				
				
				
				
				
				
				
				//THIS IS FOR FUTURE AUSTIN... THERE IS AN ISSUE... WE NEED THE LOOPS TO DO 10 TRAINING SETS THEN CHANGE MINIBATCHES... SO 0,1,2,3,4,5,6,7,8,9... 0,1,2,3... etc etc
				//The issue is that the zCompensator is running after only 2 training sets which is wrong and throws it to become 10 it needs to only change to 10 ONCE we looped through
				//all the initial training sets first so I DO 10 SETS then say ok make my zCompensator 10 etc etc
				//
				
				
				
				
				
				
				
				
				
				
				
				
				
				//Counts total number of digits
				int totalZeros = 0;
				int totalOnes = 0;
				int totalTwos = 0;
				int totalThrees = 0;
				int totalFours = 0;
				int totalFives = 0;
				int totalSixs = 0;
				int totalSevens = 0;
				int totalEights = 0;
				int totalNines = 0;
				
				//Counts total number of correct digits
				int totalCorrectlyGuessedZeros = 0;
				int totalCorrectlyGuessedOnes = 0;
				int totalCorrectlyGuessedTwos = 0;
				int totalCorrectlyGuessedThrees = 0;
				int totalCorrectlyGuessedFours = 0;
				int totalCorrectlyGuessedFives = 0;
				int totalCorrectlyGuessedSixs = 0;
				int totalCorrectlyGuessedSevens = 0;
				int totalCorrectlyGuessedEights = 0;
				int totalCorrectlyGuessedNines = 0;
				
				int totalCorrectlyGuessedDigits = 0;
				double Accuracy = 0;
				int totalDigits = 0;
				
				//Checks if 0 is the current number
				int desiredIsZero = 0;
				int desiredIs = 0;
				
				double maxElementIndex = -1;
				
			
				//BIG BOY LOOP THAT DOES IT ALL
				//LOOP FOR EPOCHS 1
				for (int i = 0; i < numOfEpochs; i++)
				{
			
					//RESETTING THE COUNTS
					 totalZeros = 0;
					 totalOnes = 0;
					 totalTwos = 0;
					 totalThrees = 0;
					 totalFours = 0;
					 totalFives = 0;
					 totalSixs = 0;
					 totalSevens = 0;
					 totalEights = 0;
					 totalNines = 0;
					
					//RESETTING THE COUNTS
					 totalCorrectlyGuessedZeros = 0;
					 totalCorrectlyGuessedOnes = 0;
					 totalCorrectlyGuessedTwos = 0;
					 totalCorrectlyGuessedThrees = 0;
					 totalCorrectlyGuessedFours = 0;
					 totalCorrectlyGuessedFives = 0;
					 totalCorrectlyGuessedSixs = 0;
					 totalCorrectlyGuessedSevens = 0;
					 totalCorrectlyGuessedEights = 0;
					 totalCorrectlyGuessedNines = 0;
					 
					 totalCorrectlyGuessedDigits = 0;
					 Accuracy = 0;
					 totalDigits = 0;
					 
					 miniBatchCompensator = 0;
					 maxElementIndex = -1;
					 
					//loop for each minibatch
					for (int j = 0; j < numOfMinibatches; j++)
					{
						//System.out.println("I am within the outer loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//loop for each training loop
						//System.out.println("K should have gone to 10");
						//System.out.println("Comp is : " + miniBatchCompensator);
						
						
						//ERROR: THIS LOOP IS GOING 6000 TIMES... which makes sense it should but the loop below isnt incrementing that at all.... 
						//			essientially its only counting 10 digits then saying eh who cares... FIX THE MINI BATCH COMPENSATOR IS 0 INITIALLY THIS IS WHY
						
						
						for (int k = miniBatchCompensator; k < numOfTrainingSetsPerMinibatch + miniBatchCompensator; k++)
						{
			
			
					
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				FORWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			
							// TODO Auto-generated method stub
							//1. Sigmoid activation function = 1/(1+e^(-z))
							//		---e is the Eulers number = 2.71828
							//		---z is the DOT PRODUCT of the vector W with vector X plus the bias
							//		---vector W is a vector containing all the weights
							//		---vector X is a vector containing all the inputs
							//		---bias is the assigned bias
							//
							//		To do this we need to first have a way to set the weights and inputs
							//		
							//		Then we can feed those vectors into a method that finds the Z
							//
							//		Then we can insert Z into the SAF in order to spit out the activation value
							// 		
							//		Then we run the results back through
							
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							//System.out.println(zMiniBatchCompensator);
							//sets the first input set and uses it below
							
							currentHiddenLayerInputSet =  hiddenLayerInputVector[k].clone();
							
						
							//System.out.println("I am within the INNER loop... zMiniBatchCompensator is " + zMiniBatchCompensator);
						//****CURRENT ERROR IS HERE SOMETHING GOES OUT OF BOUNDS OF 10 *********
							//currentHiddenLayerZSets uses the zMiniBatchCompensator because k is situated in a way to go 0 1 2 3... while the max zsets we can hold are only 0 1
							currentHiddenLayerZSets[zMiniBatchCompensator] = zAssignment(hiddenLayerWeightVector, currentHiddenLayerInputSet, hiddenLayerBiasVector);
							
							//Assigns the outcome of throwing the currentHiddenLayerZSets through the sigmoidal function
							currentHiddenLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentHiddenLayerZSets[zMiniBatchCompensator]);
					
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Grabs the current input set we acquired from the activation sums of the previous layer
							currentOutputLayerInputSet = currentHiddenLayerActivationSumSets[zMiniBatchCompensator].clone();
							
							//Throws the new input set through the zAssignment function and grabs the output layer Z sets from the equation
							currentOutputLayerZSets[zMiniBatchCompensator] = zAssignment(outputLayerWeightVector, currentOutputLayerInputSet, outputLayerBiasVector);
												
							//Temporarily stores the output layer activation sums after going through the sigmoidal function
							currentOutputLayerActivationSumSets[zMiniBatchCompensator] = sigmoidalFunction(currentOutputLayerZSets[zMiniBatchCompensator]);
			
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							// 				BACKWARD PASS          						    															  //
							////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
							
							////////////////////////////////////////////////////////////////
							// 				OUTPUT LAYER								  //
							////////////////////////////////////////////////////////////////
							
							
							//Next is to create the back propagation things that HAVE SEPERATE EQUATIONS FOR INTENRAL AND EXTERNAL LAYERS !!!!!
							//The equation for the error/ THE BIASGRADIENT in the output layer is L = (a-y) * a * (1-a) 
							// a being the outputLayerActivationSum
							//y being the desired activation value
							
							//Method that grabs the current desired input sets to use later within the loop
							currentOutputLayerTrainingSets[zMiniBatchCompensator] = outputLayerTrainingVector[k].clone();
							
							//SO THE TABLE SHOWS A CONSISTENT PATTERN WHICH ISNT GOOD IT SHOULD BE COMPLETELY RANDOM...
							//Increments the corresponding numbers aka if this is the correct output and the total
							
							//Max element in the list
							//System.out.println("Zmini is " + zMiniBatchCompensator);
							maxElementIndex = maxElementInArray(currentOutputLayerActivationSumSets[zMiniBatchCompensator]);
							desiredIsZero = 1;
							desiredIs = -1;
							
							
							
							
							//If its desired is 1 then increment total, then if guess is max of the set then also increment it
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][1] == 1.0)
							{
								totalOnes = 1 + totalOnes;
								if (maxElementIndex == 1)
								{
									totalCorrectlyGuessedOnes = 1 + totalCorrectlyGuessedOnes;
								}
								desiredIsZero = 0;
								desiredIs = 1;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][2] == 2.0)
							{
								totalTwos = 1 + totalTwos;
								if(maxElementIndex == 2)
								{
									totalCorrectlyGuessedTwos = 1 + totalCorrectlyGuessedTwos;
								}
								desiredIsZero = 0;
								desiredIs = 2;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][3] == 3.0)
							{
								totalThrees = 1 + totalThrees;
								if(maxElementIndex == 3)
								{
									totalCorrectlyGuessedThrees = 1 + totalCorrectlyGuessedThrees;
								}
								desiredIsZero = 0;
								desiredIs = 3;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][4] == 4.0)
							{
								totalFours = 1 + totalFours;
								if(maxElementIndex == 4)
								{
									totalCorrectlyGuessedFours = 1 + totalCorrectlyGuessedFours;
								}
								desiredIsZero = 0;
								desiredIs = 4;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][5] == 5.0)
							{
								totalFives = 1 + totalFives;
								if(maxElementIndex == 5)
								{
									totalCorrectlyGuessedFives = 1 + totalCorrectlyGuessedFives;
								}
								desiredIsZero = 0;
								desiredIs = 5;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][6] == 6.0)
							{
								totalSixs = 1 + totalSixs;
								if(maxElementIndex == 6)
								{
									totalCorrectlyGuessedSixs = 1 + totalCorrectlyGuessedSixs;
								}
								desiredIsZero = 0;
								desiredIs = 6;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][7] == 7.0)
							{
								totalSevens = 1 + totalSevens;
								if(maxElementIndex == 7)
								{
									totalCorrectlyGuessedSevens = 1 + totalCorrectlyGuessedSevens;
								}
								desiredIsZero = 0;
								desiredIs = 7;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][8] == 8.0)
							{
								totalEights = 1 + totalEights;
								if(maxElementIndex == 8)
								{
									totalCorrectlyGuessedEights = 1 + totalCorrectlyGuessedEights;
								}
								desiredIsZero = 0;
								desiredIs = 8;
							}
							
							if(currentOutputLayerTrainingSets[zMiniBatchCompensator][9] == 9.0)
							{
								totalNines = 1 + totalNines;
								if(maxElementIndex == 9)
								{
									totalCorrectlyGuessedNines= 1 + totalCorrectlyGuessedNines;
								}
								desiredIsZero = 0;
								desiredIs = 9;
							}
							
							if(desiredIsZero == 1)
							{
								totalZeros = 1 + totalZeros;
								if(maxElementIndex == 0)
								{
									totalCorrectlyGuessedZeros = 1 + totalCorrectlyGuessedZeros;
								}
								desiredIs = 0;
							}
							
							if(desiredIs != maxElementIndex)
							{
								pressEnterToContinue();
								//Visually shows the current image and the guessed and desired values
								printVisualExample(currentHiddenLayerInputSet,maxElementIndex, desiredIs);
								
								pressEnterToContinue();
							}

							
							maxElementIndex = -1;
									
						
							//Method that temporarily grabs the bias gradients for later use in the loops
							currentOutputLayerBiasGradientSets[zMiniBatchCompensator] = outputLayerBiasGradientFunction(currentOutputLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerTrainingSets[zMiniBatchCompensator]);
							
							//After the bias gradient we would need to then find the outputWeightGradient which is the previous's layer activation sum * the gradient bias
							currentOutputLayerWeightGradientSets[zMiniBatchCompensator] = outputLayerWeightGradientFunction(currentOutputLayerBiasGradientSets[zMiniBatchCompensator], currentHiddenLayerActivationSumSets[zMiniBatchCompensator]);
							
							////////////////////////////////////////////////////////////////
							// 				HIDDEN LAYER								  //
							////////////////////////////////////////////////////////////////
							
							//Temporarily stores the current hidden layer bias gradient for later use in revising the biases
							currentHiddenLayerBiasGradientSets[zMiniBatchCompensator] = hiddenLayerBiasGradientFunction( outputLayerWeightVector, currentHiddenLayerActivationSumSets[zMiniBatchCompensator], currentOutputLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//Gathers the hidden layers weight gradient to use in the future		
							currentHiddenLayerWeightGradientSets[zMiniBatchCompensator] = hiddenLayerWeightGradientFunction(currentHiddenLayerInputSet, currentHiddenLayerBiasGradientSets[zMiniBatchCompensator]);
							
							//used to alternate between the 10 training sets that we use
							zMiniBatchCompensator = 1 + zMiniBatchCompensator;
						}
						zMiniBatchCompensator = 0;
						
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						//REVISING BIASES AND WEIGHTS          						    															  //
						////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
						
						//Revised weights for the hidden layer TILES BF26 TO BI30
						revisedHiddenLayerWeightVector = RevisedWeightVectorFunction(currentHiddenLayerWeightGradientSets, hiddenLayerWeightVector);
						
						//Revised weights for the output layer TILES BF35 TO BH36
						revisedOutputLayerWeightVector = RevisedWeightVectorFunction(currentOutputLayerWeightGradientSets, outputLayerWeightVector);
						
						//Revised biases for the hidden layer tiles BD27 to BD29
						revisedHiddenLayerBiasVector = RevisedBiasVectorFunction(currentHiddenLayerBiasGradientSets, hiddenLayerBiasVector); 
						
						//Revised biases for the output layer tiles BD35 to BD36
						//double[] outputLayerRevisedBiasVector = RevisedBiasVectorFunction(outputLayerBiasGradientTrainingCase2, outputLayerBiasGradient, outputLayerBiasVector);
						revisedOutputLayerBiasVector = RevisedBiasVectorFunction(currentOutputLayerBiasGradientSets, outputLayerBiasVector);
						
						
						
						
			
						
						
						
						
						
						
						
						
						
						
						//SETTING THE NEW WEIGHTS AND BIASES
						
						//setting new revised hidden layer weights
						hiddenLayerWeightVector = revisedHiddenLayerWeightVector.clone();
						
						//setting new revised output layer weights
						outputLayerWeightVector = revisedOutputLayerWeightVector.clone();
						
						//setting new revised hidden layer biases
						hiddenLayerBiasVector = revisedHiddenLayerBiasVector.clone();
						
						//setting new revised output layer biases
						outputLayerBiasVector = revisedOutputLayerBiasVector.clone();
						
						//Used to alternate between each training set if theres 2
						miniBatchCompensator = numOfTrainingSetsPerMinibatch + miniBatchCompensator;
						//System.out.println("REVISEDminiBatchCompensator = " + miniBatchCompensator);
					}
					
					//Storing the final weights and biases to be used later
					recordedHiddenLayerWeightVector = hiddenLayerWeightVector.clone();
					recordedOutputLayerWeightVector = outputLayerWeightVector.clone();
					recordedHiddenLayerBiasVector = hiddenLayerBiasVector.clone();
					recordedOutputLayerBiasVector = outputLayerBiasVector.clone();
					
											
				}
				
				isMode7Availiable = 1;
			
						}
			//mode 7 is to save a weight and bias set
if(mode == 7)
			{
				csvHiddenWeightOutput = "";
				csvOutputWeightOutput = "";
				if(isMode7Availiable == 0)
				{
					System.out.println("You have not trained or leaded any network...");
				}
				//Saving the new weights into a CSV... [30][784]
				if(isMode7Availiable == 1)
				{

					
					
					//ERROR AROSE HERE BECAUSE EACH COMMA SEPERATES VALUES SO IF YOU HAVE A COMMA IN THE FRONT OR END OF THE STRING THEN AN EMPTY VALUE WILL EXIST IN THE LIST...
					System.out.println("Storing current network state");
					for (int x = 0; x < recordedHiddenLayerWeightVector.length; x++)
					{
						for (int y = 0; y < recordedHiddenLayerWeightVector[x].length; y++)
						{
							//Used to put commas only between elements not the start and not the ending so no null vlaues exist
							if(y != recordedHiddenLayerWeightVector[x].length-1)
								csvHiddenWeightOutput =  csvHiddenWeightOutput + (recordedHiddenLayerWeightVector[x][y] + ", ");
							else
							{
								csvHiddenWeightOutput =  csvHiddenWeightOutput + recordedHiddenLayerWeightVector[x][y];
							}
						}
						csvHiddenWeightOutput = csvHiddenWeightOutput + "\n";
					}
					
					for (int x = 0; x < recordedOutputLayerWeightVector.length; x++)
					{
						for (int y = 0; y < recordedOutputLayerWeightVector[x].length; y++)
						{
							if(y != recordedOutputLayerWeightVector[x].length-1)
								csvOutputWeightOutput =  csvOutputWeightOutput + (recordedOutputLayerWeightVector[x][y] + ", ");
							else
							{
								csvOutputWeightOutput =  csvOutputWeightOutput + recordedOutputLayerWeightVector[x][y];
							}
						}
						csvOutputWeightOutput = csvOutputWeightOutput + "\n";
					}

					//Used to write to the weight files
					File csvHiddenWeightFile = new File("E:\\ProfilingStuff\\finalreport\\NNP2AustinHampton\\RecordedHiddenWeights.csv");
					FileWriter fileWriter = new FileWriter(csvHiddenWeightFile);
					fileWriter.write(csvHiddenWeightOutput);
					fileWriter.close();
					
					File csvOutputWeightFile = new File("E:\\ProfilingStuff\\finalreport\\NNP2AustinHampton\\RecordedOutputWeights.csv");
					FileWriter fileWriter2 = new FileWriter(csvOutputWeightFile);
					fileWriter2.write(csvOutputWeightOutput);
					fileWriter2.close();
				}
			}
if(mode == 0)
			{
				scan.close();
				System.exit(0);
			}
			pressEnterToContinue();
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// 				FORWARD PASS          						    															  //
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	
	//Method to solve the sigmoidal function incorporating the Z we found before hand:
	public static double[] sigmoidalFunction(double[] z)
	{
		//Eulers number
		double e = 2.71828;
		//initializing activation sum array that will store the output of the sigmoidal function 1/(1+e^(-z))
		double[] activationSum = new double[z.length];
		
		for (int i = 0; i < z.length; i++)
		{
			activationSum[i] = 1/(1 + Math.pow(e,-z[i]));
		}
		
		//returns the activation sum array
		return activationSum;
	}
	
	//Method to find the dot product of the weightVector and the inputVector plus the bias
	public static double[] zAssignment(double[][] weightVector, double[] inputVector, double[] biasVector)
	{
		//setting up the array Z the dot product of w and i plus the bias
		double[] z = new double[biasVector.length];
		double dotProduct = 0;

		//Loops situated to 
		for (int i = 0; i < weightVector.length; i++)
		{
			dotProduct = 0;
			for (int j = 0; j < weightVector[i].length; j++)
			{

				//Dotproduct is summation of the dotproduct of wight and input + the previous summation
				dotProduct = weightVector[i][j] * inputVector[j] + dotProduct;
			}
			z[i] = dotProduct + biasVector[i];
		}
		return z;
	}

	////////////////////////////////////////////////////////////////
	// 				OUTPUT LAYER								  //
	////////////////////////////////////////////////////////////////
	
	//Method that applies the outputLayerBiasGradient Function to the previously acquired activation sum in order to get the bias gradient IE L = (a-y) * a * (1-a)
	public static double[] outputLayerBiasGradientFunction(double[] outputLayerActivationSum, double[] outputLayerTrainingValues) 
	{
		double[] outputLayerBiasGradient = new double[outputLayerActivationSum.length];
		
		for (int i = 0; i<outputLayerBiasGradient.length; i++)
		{
			outputLayerBiasGradient[i] = (outputLayerActivationSum[i]- outputLayerTrainingValues[i])*(outputLayerActivationSum[i])*(1-outputLayerActivationSum[i]);
		}
		
		return outputLayerBiasGradient;
	}
	

	//Method to get the outputLayerWeightGradient by multiplying the previouses aka the HIDDEN layer activationSums with the biasGradient we just found
	public static double[][] outputLayerWeightGradientFunction(double[] outputLayerBiasGradient, double[] hiddenLayerActivationSum)
	{
		double[][] outputLayerWeightGradient = new double[outputLayerBiasGradient.length][hiddenLayerActivationSum.length];
		
		//loop that goes through the hiddenLayers activation sums and multiplies them with the outputLayerbiasGradients
		for (int i = 0; i < outputLayerWeightGradient.length; i++)
		{
			for (int j = 0; j < outputLayerWeightGradient[i].length; j++)
			{
				outputLayerWeightGradient[i][j] =  hiddenLayerActivationSum[j] * outputLayerBiasGradient[i];
			}
		}
		return outputLayerWeightGradient;
	}


	////////////////////////////////////////////////////////////////
	// 				HIDDEN LAYER								  //
	////////////////////////////////////////////////////////////////
	
	//Method that applies the hiddenLayer Bias Gradient function using the outputlayerWeightVector, HiddenLayerActivationSums, and the outputLayerBiasGradient
	// the equation for this problem is: outputLayerWeightVector * (ActivationSum * (1-ActivationSum))
	public static double[] hiddenLayerBiasGradientFunction(double[][] outputLayerWeightVector, double[] hiddenLayerActivationSum, double[] outputLayerBiasGradient)
	{
		double[] hiddenLayerBiasGradient = new double[hiddenLayerActivationSum.length];
		double currentWeightSummation = 0;
		
		//this loop gets the summation of the weight vector then applies it to the overall equation
		for (int i = 0; i < hiddenLayerBiasGradient.length; i++)
		{
			currentWeightSummation = 0;
			
			//use this loop to get the summation of the weight vector
			for (int j = 0; j < outputLayerBiasGradient.length; j++)
			{
				currentWeightSummation = (outputLayerWeightVector[j][i] * outputLayerBiasGradient[j]) + currentWeightSummation;
			}
			
			//Now that we have the currentWeight Summation we can simply apply it to the rest of the equation
			hiddenLayerBiasGradient[i] = currentWeightSummation * (hiddenLayerActivationSum[i] * (1 - hiddenLayerActivationSum[i]));
		}
		return hiddenLayerBiasGradient;
	}

	//Method that uses the hiddenLayers input vector and multiples it with the hiddenLayerBiasGradients
	public static double[][] hiddenLayerWeightGradientFunction(double[]  hiddenLayerInputVector, double[] hiddenLayerBiasGradient)
	{
		double[][] hiddenLayerWeightGradient = new double[hiddenLayerBiasGradient.length][hiddenLayerInputVector.length];

		for (int i = 0; i < hiddenLayerInputVector.length; i++)
		{

			for (int j = 0; j < hiddenLayerBiasGradient.length; j++)
			{

				hiddenLayerWeightGradient[j][i] =  hiddenLayerInputVector[i] * hiddenLayerBiasGradient[j];

			}
		}
		
		return hiddenLayerWeightGradient;
	}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 				REVISING BIASES AND WEIGHTS          						    															  //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Method to revise the weightvector based on back propagation  W = w - (ETA / SizeOfMiniBatch) * hiddenLayerWeightGradient + hiddenLayerWeightGradientTrainingCase2
	public static double[][] RevisedWeightVectorFunction(double[][][] currentHiddenLayerWeightGradientSets, double[][] hiddenLayerWeightVector)
	{
		///////////////////////////////////////////////////////////////////////////COMAPRE WITH THE SPREADSHEET
		//weightVector
		double eta = .5;
		double sizeOfMiniBatch = 10;
		double[][] hiddenLayerRevisedWeightVector = new double[hiddenLayerWeightVector.length][hiddenLayerWeightVector[0].length];
		double summationOfWeightGradients = 0;
		
		
//NOTE: WILL MOST LIKELY HAVE TO EDIT THIS AS THIS IS RESTRICTED TO ONLY 2 TRAINING SETS PER MINI BATCH DUE TO IT TAKING IN 2 TRAINING CASES----------------------------------------------------------------------------------------------------------------------------------------------------------------
//				Best thing to do would be to change the parameters to the array containing all the weightGradients for this pass and adding another inner loop to loop through all the training sets
		for (int i = 0; i < hiddenLayerRevisedWeightVector.length; i++)
		{
			for (int j = 0; j < hiddenLayerRevisedWeightVector[i].length; j++)
			{
				summationOfWeightGradients = 0;
				//THIS LOOP HAS TO ALLOW ONE TO GET THE SUMMATION 
				for (int k = 0; k < currentHiddenLayerWeightGradientSets.length; k++)
				{
					summationOfWeightGradients = currentHiddenLayerWeightGradientSets[k][i][j] + summationOfWeightGradients;
				}
				hiddenLayerRevisedWeightVector[i][j] = hiddenLayerWeightVector[i][j] - (eta/sizeOfMiniBatch) * summationOfWeightGradients;
			}
			
		}
		return hiddenLayerRevisedWeightVector;
	}

	//Method to revise the bias vectors during back propogation: B = b - (ETA / SizeOfMiniBath) *
	public static double[] RevisedBiasVectorFunction(double[][] hiddenLayerBiasGradient, double[]  hiddenLayerBiasVector)
	{
		double eta = .5;
		double sizeOfMiniBatch = 10;
		double[] revisedBiasVector = new double[hiddenLayerBiasVector.length];
		double summationOfHiddenLayerBiasGradient = 0;
		for (int i = 0; i < hiddenLayerBiasVector.length; i++)
		{
			summationOfHiddenLayerBiasGradient = 0;
			for (int j = 0; j < hiddenLayerBiasGradient.length; j++)
			{
				summationOfHiddenLayerBiasGradient = hiddenLayerBiasGradient[j][i] + summationOfHiddenLayerBiasGradient;
			}
			revisedBiasVector[i] = hiddenLayerBiasVector[i] - (eta/sizeOfMiniBatch) * summationOfHiddenLayerBiasGradient;
		}
		
		return revisedBiasVector;
	}

	
	
	//Method to convert the CSV into proper input aka into a [60,000][784] ARRAY
	public static double[][]convertCsvToArray(String inputFilePath, int numOfMinibatches, int numOfTrainingSetsPerMinibatch, int numOfNodesInInputLayer) throws FileNotFoundException
	{
		double convertedInputSets[][] = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInInputLayer + 1];
		//parsing a CSV file into Scanner class constructor  
		Scanner sc = new Scanner(new File(inputFilePath));  
		sc.useDelimiter(",|\\n");   //sets the delimiter pattern  
		while (sc.hasNext())  //returns a boolean value  
		{  
			//Loops through how many training sets there are... 60,000
			for (int i = 0; i < convertedInputSets.length; i++)
			{
				//J's dimensions are correct it is 784 starting at 0 so thus 785 elements
				for (int j = 0; j < convertedInputSets[i].length; j++)
				{
					//Converts the next element into a double  HAVE TO DO THE DIVISION LATER DUE TO LABEL NEEDING TO BE 0-9 
					convertedInputSets[i][j] = Double.valueOf(sc.next());
				}
			}
		}
		
		//Loops through how many training sets there are... 60,000
		for (int i = 0; i < convertedInputSets.length; i++)
		{
			//J's dimensions are correct it is 784 starting at 0 so thus 785 elements
			for (int j = 1; j < convertedInputSets[i].length; j++)
			{
				//Converts the next element into a double  HAVE TO DO THE DIVISION LATER DUE TO LABEL NEEDING TO BE 0-9 
				convertedInputSets[i][j] = convertedInputSets[i][j]/255;
			}
		}
		
		sc.close();  //closes the scanner
		
		return convertedInputSets;
	}


	public static double[][]convertCsvToHiddenWeight(String inputFilePath, int numOfNodesInHiddenLayer, int numOfNodesInInputLayer) throws FileNotFoundException
	{
		//[numOfNodesInHiddenLayer][numOfNodesInInputLayer]
		double hiddenWeightSet[][] = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
		//parsing a CSV file into Scanner class constructor  
		Scanner sc = new Scanner(new File(inputFilePath));  
		sc.useDelimiter(",|\\n");   //sets the delimiter pattern  
		while (sc.hasNext())  //returns a boolean value  
		{  
			//Loops through how many training sets there are... 60,000
			for (int i = 0; i < hiddenWeightSet.length; i++)
			{
				//J's dimensions are correct it is 784 starting at 0 so thus 785 elements
				for (int j = 0; j < hiddenWeightSet[i].length; j++)
				{
					//System.out.println(j);
					//Converts the next element into a double  HAVE TO DO THE DIVISION LATER DUE TO LABEL NEEDING TO BE 0-9 
					//ERRORS OUT DUE TO NEW LINE
					hiddenWeightSet[i][j] = Double.valueOf(sc.next());
				}
			}
		}
		
		sc.close();  //closes the scanner
		
		return hiddenWeightSet;
	}
	
	public static double[][]convertCsvToOutputWeight(String inputFilePath, int numOfNodesInOutputLayer, int numOfNodesInHiddenLayer) throws FileNotFoundException
	{
		//[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
		double outputWeightSet[][] = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
		
		//parsing a CSV file into Scanner class constructor  
		Scanner sc = new Scanner(new File(inputFilePath));  
		sc.useDelimiter(",|\\n");   //sets the delimiter pattern  
		while (sc.hasNext())  //returns a boolean value  
		{  
			//Loops through how many training sets there are... 60,000
			for (int i = 0; i < outputWeightSet.length; i++)
			{
				//J's dimensions are correct it is 784 starting at 0 so thus 785 elements
				for (int j = 0; j < outputWeightSet[i].length; j++)
				{
					//Converts the next element into a double  HAVE TO DO THE DIVISION LATER DUE TO LABEL NEEDING TO BE 0-9 
					outputWeightSet[i][j] = Double.valueOf(sc.next());
				}
			}
		}
		
		sc.close();  //closes the scanner
		
		return outputWeightSet;
	}
	//Used to shuffle the indexes fo the data so that it is random and there are never duplicates
	
	public static double[][] shuffleArray(double[][] convertedInputSets)
	{
	
		double[] temp = new double[convertedInputSets.length];
		Random rand = new Random();
		
		//Loop that goes intul we hit 59999
		for (int i = 0; i < convertedInputSets.length; i++) {
			//Find a random index within the array
			int randomIndexToSwap = rand.nextInt(convertedInputSets.length);
			
			//Temporarily store the information of that spot in the array in temp
			temp = convertedInputSets[randomIndexToSwap];
			
			//Swap that random index with the current input set
			convertedInputSets[randomIndexToSwap] = convertedInputSets[i];
			
			//Put the stored array back into where the current i is...
			convertedInputSets[i] = temp;
		}
		//System.out.println(Arrays.toString(convertedInputSets));
		
		return convertedInputSets;
	}

	public static double maxElementInArray(double[] currentOutputLayerActivationSumSets)
	{
		double maxElementIndex = -1;
		double maxElement = -1;
		
		for (int i = 0; i < currentOutputLayerActivationSumSets.length; i++)
		{
			if(maxElement < currentOutputLayerActivationSumSets[i])
			{
				maxElement = currentOutputLayerActivationSumSets[i];
				maxElementIndex = i;
			}
			
		}
		
		return maxElementIndex;
	}
	
	
	//Grabber methods
	
	//This is going to be where we input aka our PIX stuff...
	public static double[][] inputVectorGrabber(double[][]  convertedInputSets)
	{
		double[][]  inputVector = new double[convertedInputSets.length][convertedInputSets[0].length];
		//InputVector[6000*10][784];
		
		//Loop that skips the first of every row as it is the label and grabs the actual input
		for (int i = 0 ; i < inputVector.length; i++)
		{
			for (int j = 1; j < inputVector[i].length; j++)
			{
			inputVector[i][j] = convertedInputSets[i][j];
			}
		}
		
		return inputVector;
	}
	
	public static double[][] hiddenLayerWeightVectorGrabber(int numOfNodesInHiddenLayer, int numOfNodesInInputLayer)
	{

		//[30][784]
		double[][]  hiddenLayerWeightVector = new double[numOfNodesInHiddenLayer][numOfNodesInInputLayer];
		
		for (int i = 0; i < hiddenLayerWeightVector.length; i++)
		{
			for(int j = 0; j < hiddenLayerWeightVector[i].length; j++)
			{
				//Produces a weight between -1 and 1
				hiddenLayerWeightVector[i][j] = (Math.random() * 2) - 1;
			}
		}
		
		
		return hiddenLayerWeightVector;
	}

	public static double[] hiddenLayerBiasVectorGrabber(int numOfNodesInHiddenLayer)
	{
		double[] hiddenLayerBiasVector = new double[numOfNodesInHiddenLayer];
		
		for(int i = 0; i < hiddenLayerBiasVector.length; i++)
		{
			//Produces a weight between -1 and 1
			hiddenLayerBiasVector[i] = (Math.random() * 2) - 1;
		}
		
		//returns weights
		return hiddenLayerBiasVector;
	}

	public static double[][] outputLayerWeightVectorGrabber(int numOfNodesInOutputLayer,int numOfNodesInHiddenLayer)
	{
		//[10][30]
		double[][] outputLayerWeightVector = new double[numOfNodesInOutputLayer][numOfNodesInHiddenLayer];
		
		for (int i = 0; i < outputLayerWeightVector.length; i++)
		{
			for(int j = 0; j < outputLayerWeightVector[i].length; j++)
			{
				//Produces a weight between -1 and 1
				outputLayerWeightVector[i][j] = (Math.random() * 2) - 1;
			}
		}
		
		return outputLayerWeightVector;
	}
	
	public static double[] outputLayerBiasVectorGrabber(int numOfNodesInOutputLayer)
	{
		//setting size
		double[] outputLayerBiasVector = new double[numOfNodesInOutputLayer];
		
		for(int i = 0; i < outputLayerBiasVector.length; i++)
		{
			//Produces a weight between -1 and 1
			outputLayerBiasVector[i] = (Math.random() * 2) - 1;
		}
		
		//returns weights
		return outputLayerBiasVector;
		
		
	}

	
	
	//This is going to be where we put the LABELs
	public static double[][] outputLayerTrainingVectorGrabber(int numOfNodesInOutputLayer, int numOfMinibatches, int numOfTrainingSetsPerMinibatch, double[][] convertedInputSets)
	{
		//outputLayerTrainingVector[60000][10]
		double[][] outputLayerTrainingVector = new double[numOfMinibatches * numOfTrainingSetsPerMinibatch][numOfNodesInOutputLayer];

		//This loops 60000 times
		for(int i = 0; i < outputLayerTrainingVector.length; i++)
		{
			//this loops 10 times and it only spits out a 1 whenever j == the desired output aka the label
			for(int j = 0; j < outputLayerTrainingVector[i].length; j++)
			{
		
				outputLayerTrainingVector[i][j] = 0;
				if((double) j == convertedInputSets[i][0])
				{
					//System.out.println("True J = " + j);
					//System.out.println("Blah Input = " + convertedInputSets[i][0]);
					outputLayerTrainingVector[i][j] = (double) j;
				}			
			}
		}
		
		
		return outputLayerTrainingVector;
	}
	
	
	public static void pressEnterToContinue() throws IOException
	{ 
	       System.out.println("Press Enter key to continue...");
	       {
	           System.in.read();
	       }  

	}

	public static void printVisualExample(double[] currentHiddenLayerInputSet, double maxElementIndex, double desiredIs)
	{
		System.out.println("######################################################################################################################");
		System.out.println("Desired input = " + desiredIs + "                                     Current guess = " + maxElementIndex);
		if(desiredIs == maxElementIndex)
		{
			System.out.println("Correct Guess");
		}
		else
			System.out.println("Incorrect Guess");

		//loop the goes through each line and prints a # if theres anything there (ROUGH DRAFT)
		for (int i = 0; i < currentHiddenLayerInputSet.length; i++)
		{
			if(i%28 == 0)
				System.out.println();
			if(currentHiddenLayerInputSet[i] > .75)
			{
				System.out.print("#");
			}
			if(.75 > currentHiddenLayerInputSet[i] &&  currentHiddenLayerInputSet[i]> .5)
			{
				System.out.print("▓");
			}
			if(.5 > currentHiddenLayerInputSet[i] &&  currentHiddenLayerInputSet[i]> .25)
			{
				System.out.print("▒");
			}
			if(.25 > currentHiddenLayerInputSet[i] &&  currentHiddenLayerInputSet[i]> 0)
			{
				System.out.print("░");
			}
			else
				System.out.print(" ");
		}
	}
	
}