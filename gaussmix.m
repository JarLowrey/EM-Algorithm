%PLEASE FUCKING WORK

function gaussmix(numClusters, dataFile,modelFile)
    numClustersNumeric = str2double(numClusters);
    [data,numExamples,numFeatures] = scanIn(dataFile);
    [means, variances, priors] = init(data,numClustersNumeric);
    logProb = -realmax;
    
    repeat=true;
    iterationNo=0;
    while repeat %&& iterationNo<2
        [clusterLogDist,clusterLogDistDenominators] = eStep(data,numExamples,numFeatures,numClustersNumeric,means,variances,priors);
        clusterLogDist
        clusterLogDistDenominators
        [means, variances, priors] = mStep(data,numExamples,numFeatures,numClustersNumeric,clusterLogDist);
        means
        variances(1,:,:)
        priors
        probAfterIteration = totalLogLikelihoodOfData(numExamples,clusterLogDistDenominators) ;       
        repeat = ( probAfterIteration-logProb ) > 0.001;
        logProb = probAfterIteration;
        %difference = probAfterIteration-logProb
        iterationNo = iterationNo+1
    end
    
    writeOutput(modelFile,clusterLogDist,data);
end


function [rawData,numExamples,numFeatures] = scanIn( dataFile)
    fid = fopen(dataFile,'r'); % Open text file
    
    exAndFeat = cell2mat( textscan(fid,'%d %d',1) );  % Read first line
    numExamples = exAndFeat(1); 
    numFeatures = exAndFeat(2);
    
    rawData = cell2mat( textscan(fid,repmat('%f ',[1,numFeatures])) ); %textscan repeats until it finds differentm format
    
    fclose(fid);
end

function writeOutput(modelFile,clusterLogDist,data)
    fid = fopen(modelFile,'w'); % Open text file
    
    exAndFeat = size(data);
    numExamples = exAndFeat(1);
    numFeatures = exAndFeat(2);
    exAndClus = size(clusterLogDist);
    numClusters = exAndClus(2);
    
    %find the max clusterLogDistribution of each example and save it as the
    %cluster the example is assign to
    assignedCluster = (1:numExamples).*0;
    for ex=1:numExamples
        max = -realmax;
        
        for c=1:numClusters
            if(clusterLogDist(ex,c)>max)
                assignedCluster(ex) = c;
                max = clusterLogDist(ex,c);
            end
        end
        
        fprintf(fid,'%d ',assignedCluster(ex) );
        fprintf(fid,repmat('%f ',[1,numFeatures]),data(ex,:) );
        fprintf(fid,'\n');
    end
        
    fclose(fid);
end

function [means, variances, priors] = init(data, numClusters)
    %initialize cluster priors to a uniform distribution
    priors( 1:numClusters ) =  1 / numClusters;
    
    %initialize priors for each cluster to a uniform
    %distribution
    numExAndFeat = size(data);
    numEx = numExAndFeat(1);
    numFeat =numExAndFeat(2);
    
    %{
    %init cluster distributions to a uniform distribution, but take the log
    %since that is expected
    clusterLogDist = zeros(numEx,numClusters);
    for i=1:numEx
        for j=1:numClusters
            clusterLogDist(i,j) = log( priors(j) );
        end
    end
    %}
    
    %find mins and maxs of each data feature
    mins( 1 : numFeat ) =  realmax;
    maxs( 1 : numFeat ) =  -realmax;
    for i=1:numEx
       for j=1:numFeat
           if(data(i,j)>maxs(j))
               maxs(j)= data(i,j) ;               
           end
           if(data(i,j)<mins(j))
               mins(j)=data(i,j);
           end
       end
    end
    
    %init means to a random value in the range of the data
    %init variances to a diagonal matrix of range/2
    means = zeros(numClusters,numFeat);
    variances = zeros(numClusters,numFeat,numFeat);
    
    for i=1:numClusters
        for j=1:numFeat
            range = maxs(j)-mins(j);
            %means(i,j) = range*rand()+mins(j);
            means(i,j)=mins(j);
            variances(i,j,j) = range / 2;
        end
    end
    
    %debug-print to console
    %{
    means
    priors
    variances(1,:,:)%ensure that variances are diagonal
    %}
end

function [clusterLogDist,clusterLogDistDenominators] = eStep(data,numExamples,numFeatures,numClusters,means,variances,priors)
    clusterLogDist = zeros(numExamples,numClusters);
    clusterLogDistDenominators = (1:numExamples).*0;
    clusterLogDistDenominators = double(clusterLogDistDenominators);
    
    for ex=1:numExamples
        logPMax = -realmax;
        
        for c=1:numClusters
            %numerator is ln ( P(Ci) * P(Xk | Ci) )
            %P(Xk | Ci) = multivariate normal distribution
            %http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
            %apply log to that function, the coeffecient is now added and
            %the exponent of e is now subtracted
            clusterVariance =  reshape ( variances(c,:,:),[numFeatures numFeatures] ) ;
            normalLogCoeff = -.5 * log( (2*pi)^double(numExamples) * det( clusterVariance ) ); %2pi^ex or numExamples?
            %as my vectors are row vectors instead of column vectors, the
            %transpose has switched
            %NOTE:inverse tempVariance completed via the divide-faster than
            %inv(tempVariance)
            normalLogOfExp = -.5 * ( data(ex,:) - means(c,:) ) / clusterVariance *  transpose( data(ex,:) - means(c,:) ) ; 
                        
            %P(Ci) is just from the prior. Add the ln values to get
            %numerator
            clusterLogDist(ex,c) = log(priors(c)) + normalLogCoeff + normalLogOfExp; 
            
            if(  clusterLogDist(ex,c) > logPMax )
                logPMax = clusterLogDist(ex,c) ;
            end
        end
        
        %after finding all the numerators & logPMax, use LogSum over the
        %numerators to find the denominator (normalizing constant) of every
        %example
        logSum=0;
        for c=1:numClusters
            logSum = logSum + exp(clusterLogDist(ex,c)-logPMax);
        end
        if(ex==10)
            logPMax
            logSum
        end
        clusterLogDistDenominators(ex) = logPMax + log(logSum);
        
        %Once denominator has been found, subtract it from this example's
        %numerators
        clusterLogDist(ex,:) = clusterLogDist(ex,:) - clusterLogDistDenominators(ex);
    end
    
    %now we have the cluster distributions. The distributions indicate the
    %weight each data point has towards each cluster
end

function [means, variances, priors] = mStep(data,numExamples,numFeatures,numClusters,clusterLogDist)
    %NOTE :safe to convert from clusterLogDist back to probability as the value
    %should be large
    
    %for each cluster prior, average the distribution over every examples
    %prior(cluster=c) = 
    % [ SUM OVER DATA EXAMPLES Prob(c|data) ]/numDataExamples
    priors = (1:numClusters).*0;
    priors = double(priors);
    
    for c=1:numClusters
        for ex=1:numExamples
            priors(c) = priors(c) + exp(clusterLogDist(ex,c));
        end
        priors(c) = priors(c) / numExamples;
    end
    
    %for each mean, find weighted average of the data
    %MEAN(cluster=c) = [ SUM OVER DATA EXAMPLES data * Prob(c|data) ] / [SUM OVER DATA EXAMPLES Prob(c|data) ]
    means = zeros(numClusters,numFeatures);
    for c=1:numClusters
        denom=0;
        for ex=1:numExamples
            %for each cluster mean, sum up the data features weighted by
            %probability
            weightProb = exp(clusterLogDist(ex,c));
            means(c,:) = means(c,:) + ( data(ex,:) .* weightProb );
            %sum the weights to find the denominator
            denom = denom + weightProb;
        end
        means(c,:) = means(c,:) ./ denom;
    end
    
    %for each variance, find distance from mean, square, and weight by dist
    %variance(cluster=c) = [SUM OVER DATA EXAMPLES (data-mean^2) * Prob(c|data)] / [SUM OVER DATA EXAMPLES Prob(c|data) ]
    variances = zeros(numClusters,numFeatures,numFeatures);
    for c=1:numClusters
        
        varianceDiagonal = (1:numFeatures).*0;
        varianceDiagonal = double(varianceDiagonal);
        denom=0;
        for ex=1:numExamples
            %for every data feature, find sqaure distance from mean and
            %multiply by probability weight
            weightProb = exp(clusterLogDist(ex,c));
            squareDistFromMean = ( data(ex,:) - means(c,:) ).^2;
            varianceDiagonal = squareDistFromMean.*weightProb;
            %sum the weights to find the denominator
            denom = denom + weightProb;
        end
        
        varianceDiagonal = varianceDiagonal ./ denom;
        
        %assign the variances to the diagonal of the variance cluster matrix
        for f=1:numFeatures
            variances(c,f,f) = varianceDiagonal(f);
        end
    end
end

function totalLogProb = totalLogLikelihoodOfData(numExamples,clusterLogDistDenominators)
    %totalLogProb=0; 
    
    %{
    %logsum over all numerators
    numExAndClust=size(clusterLogDistNumerators);
    numExamples = numExAndClust(1);
    numClusters = numExAndClust(2);
    
    for ex=1:numExamples
       logSum=0;
       for c=1:numClusters
           logSum = logSum+exp(clusterLogDistNumerators(ex,c) - logPMax);
       end
       logSum = logPMax + log(logSum);
       
       totalProb = totalProb + logSum;
    end
    %}
    
    %{
    % straight sum the denominators to find the total    
    for ex=1:numExamples
        totalLogProb = totalLogProb + clusterLogDistDenominators(ex);
    end
    %}
    
    %logsum the denominators to find the total  
    logPMax=-realmax;
    for ex=1:numExamples
        if clusterLogDistDenominators(ex) > logPMax
            logPMax = clusterLogDistDenominators(ex);
        end
    end
    
    logSum=0;
    for ex=1:numExamples
        logSum = logSum + exp(clusterLogDistDenominators(ex) - logPMax);
    end
    
    totalLogProb = logPMax + log(logSum);
end
























