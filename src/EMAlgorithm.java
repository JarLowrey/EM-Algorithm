import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;


public class EMAlgorithm {
	private static double[][] rawData;
	private static int[] clusterAssignments;//cluster assigned can be 0 through numClusters-1
	private static int numExamples,numFeatures,numClusters;
	
//	RUN WITH 	./gaussmix <\# of cluster> <data file> <model file>
	public static void main(String args[]) throws IllegalArgumentException, IOException{
		if(args.length!=5){throw new IllegalArgumentException("wrong number of arguments");}
		
		readFile(args[2]);
		
		numClusters = Integer.parseInt(args[1]);
		clusterAssignments=new int[numExamples];
		
		//BUSINESS LOGIC
		
		writeFile(args[3]);
	}
	
	/**
	 * Initialize means, cluster priors ( to a uniform distribution ) and
	 * the standard deviations ( to a fixed fraction of the range of each variable )
	 */
	private static void init(){
		
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