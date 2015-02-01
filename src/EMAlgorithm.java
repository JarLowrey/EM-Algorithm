import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

import org.apache.commons.math3.distribution.NormalDistribution;

public class EMAlgorithm {
	private static double[][] rawData,stdDevs,means;
	private static double[] priorClusterProb;
	private static int[] clusterAssignments;//cluster assigned can be 0 through numClusters-1
	private static int numExamples,numFeatures,numClusters;
	
//	RUN WITH 	./gaussmix <\# of cluster> <data file> <model file>
	public static void main(String args[]) throws IllegalArgumentException, IOException{
		if(args.length!=5){throw new IllegalArgumentException("wrong number of arguments");}
		if( ! args[0].equalsIgnoreCase("gaussmix")){throw new IllegalArgumentException("wrong starting argument");}

		numClusters = Integer.parseInt(args[1]);
		readFile(args[2]);
		init();
		
		//BUSINESS LOGIC
		eStep();
		mStep();
		
		//print out clustered data
		writeFile(args[3]);
	}
	
	private static void eStep(){
		double featureNumerators[][][]=new double[numClusters][numExamples][numFeatures];
		double featureDenominators[][]=new double[numExamples][numFeatures];
		
		double biggestNumerator=0;//is there only 1 biggest numerator, or a biggestNumerator per feature????
		
		//find the numerators
		for(int i=0;i<numClusters;i++){
			for(int j=0;j<numExamples;j++){
				for(int k=0;k<numFeatures;k++){
					NormalDistribution norm = new NormalDistribution(means[i][k],stdDevs[i][k]);
					final double probFeatureGivenCluster = norm.density(rawData[j][k]);
					//for every feature in every data point, there will be a numerator 
					featureNumerators[i][j][k] = Math.log( priorClusterProb[i] ) + Math.log(probFeatureGivenCluster);
					
					if(featureNumerators[i][j][k]>biggestNumerator){biggestNumerator=featureNumerators[i][j][k];}
				}
			}
		}

		//find the  denominators
		for(int i=0;i<numClusters;i++){
			for(int j=0;j<numExamples;j++){
				for(int k=0;k<numFeatures;k++){
					//the denominator is independent of the clusters, and is essentially a normalizing factor
					//denominator is found by summing up each feature's total across the different clusters
					//log sum trick is used here
					featureDenominators[j][k] += Math.exp( featureNumerators[i][j][k]-biggestNumerator );
				}
			}
		}
		
		//iterate over every feature of an example to find probability of example belonging to a cluster
		double probExampleBelongsToCluster[][] = new double[numClusters][numExamples];
		for(int i=0;i<numClusters;i++){
			for(int j=0;j<numExamples;j++){
				for(int k=0;k<numFeatures;k++){
					probExampleBelongsToCluster[i][j] += featureNumerators[i][j][k] - Math.log( featureDenominators[j][k] );
				}
			}
		}
		
		//assign examples to cluster with greatest probability
		for(int i=0;i<numExamples;i++){
			double max=Double.MIN_VALUE;
			for(int j=0;j<numClusters;j++){
				if(probExampleBelongsToCluster[j][i]>max){
					max=probExampleBelongsToCluster[j][i];
					clusterAssignments[i]=j;
				}
			}
		}
		
	}
	
	private static void mStep(){
		//reset mean and stdDev
		means=new double[numClusters][numFeatures];
		stdDevs=new double[numClusters][numFeatures];
		
		//sum up data features and track how many data examples are in each cluster
		int[] numExamplesInEachCluster=new int[numClusters];
		for(int e=0;e<numExamples;e++){
			for(int f=0;f<numFeatures;f++){
				numExamplesInEachCluster [ clusterAssignments[e] ] ++;
				means[ clusterAssignments[e] ][f] += rawData[e][f];
			}
		}

		//divide each mean by number of examples with a given cluster
		for(int c=0;c<numClusters;c++){
			for(int f=0;f<numFeatures;f++){
				means[c][f] /= numExamplesInEachCluster[c];
			}
		}

		//sum up distances of data from new mean
		for(int e=0;e<numExamples;e++){
			for(int f=0;f<numFeatures;f++){
				stdDevs[ clusterAssignments[e] ][f] += Math.pow(rawData[e][f] - means[ clusterAssignments[e] ][f],2 );
			}
		}
		
		//divide each stDev by number of examples in a given cluster, then set std to sqrt of that value
		for(int c=0;c<numClusters;c++){
			for(int f=0;f<numFeatures;f++){
				stdDevs[c][f]/=numExamplesInEachCluster[c];
				stdDevs[c][f] = Math.sqrt( stdDevs[c][f] );
			}
		}
		
		//assign priors
		for(int c=0;c<numClusters;c++){
			priorClusterProb[c] = 1/numExamplesInEachCluster[c];
		}
	}
	
	
	/**
	 * Initialize means, cluster priors ( to a uniform distribution ) and
	 * the standard deviations ( to a fixed fraction of the range of each variable )
	 */
	private static void init(){
		//find the min and max of each data feature
		double min[] = new double[numFeatures];
		double max[] = new double[numFeatures];
		
		for(int i=0;i<numFeatures;i++){
			min[i]=Double.MAX_VALUE;
			max[i]=Double.MIN_VALUE;
			for(int j=0;j<numExamples;j++){
				double currValue=rawData[j][i];
				if(currValue<min[i]){
					min[i]=currValue;
				}
				if(currValue>max[i]){
					max[i]=currValue;
				}
			}
		}
		
		//assign means and standard deviations based on range (max-min) of data 
		means=new double[numClusters][numFeatures];
		stdDevs=new double[numClusters][numFeatures];
		for(int j=0;j<numClusters;j++){
			for(int i=0;i<numFeatures;i++){
				final double diff=max[i]-min[i];
				means[j][i]= diff*Math.random()+min[i];//initialize to a random point in the range of data
				stdDevs[j][i]=diff/2;//a fixed fraction of the range of each variable
			}
		}
		
		//assign priors to uniform
		priorClusterProb= new double[numClusters];
		for(int i=0;i<numClusters;i++){
			priorClusterProb[i]=1.0/numClusters;//uniform distribution
		}
	}
	
	/**
	 * Set numExamples, numFeatures, and write all features into the data[][] array
	 * @param inputFileName
	 * @throws IOException
	 * @throws IllegalArgumentException
	 */
	private static void readFile(String inputFileName) throws IOException,IllegalArgumentException {
		File inputFile = new File(inputFileName);
		FileInputStream fis = new FileInputStream(inputFile);
	 
		//Construct BufferedReader from InputStreamReader
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
	 
		//get the first line parsed
		String firstLine = br.readLine();
		String[] numExamplesAndFeatures = firstLine.split(" ");
		numExamples = Integer.parseInt(numExamplesAndFeatures[0]);
		numFeatures = Integer.parseInt(numExamplesAndFeatures[1]);
		
		rawData = new double[numExamples][numFeatures];
		clusterAssignments=new int[numExamples];
		
		//parse all features into the data[][] array
		for(int example=0;example<numExamples;example++){
			String line = br.readLine();
			String[] fileData = line.split(" ");
			
			if(fileData.length!=numFeatures){
				br.close();
				throw new IllegalArgumentException("wrong number of features in data file");
			}
			
			for(int feature=0;feature<numFeatures;feature++){
				rawData[example][feature]=Double.parseDouble(fileData[feature]);
			}
		}
		
		br.close();
	}

	/**
	 * print out data with cluster assignment numbers
	 * @param outputFileName
	 * @throws IOException
	 * @throws IllegalArgumentException
	 */
	private static void writeFile(String outputFileName) throws IOException,IllegalArgumentException{
		PrintWriter writer = new PrintWriter(outputFileName, "UTF-8");

		writer.write( numClusters+ " ");
		writer.write("numFeatures\n");
		
		//loop through clusterAssignments, and print out data and cluster # if it matches current cluster.
		//(Poor performance, but easy to understand)
		for(int cluster=0;cluster<numClusters;cluster++){
			for(int example=0;example<numExamples;example++){
				if(clusterAssignments[example]==cluster){
					writer.write( (cluster+1) + " ");//cluster #
					for(int feature=0;feature<numFeatures;feature++){//data
						writer.write(rawData[example][feature]+" ");
					}
					writer.write("\n");//newline
				}
			}
		}
		
		writer.close();
	}
}