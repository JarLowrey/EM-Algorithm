

function gaussmix(numClusters, dataFile,modelFile)
    numClustersNumeric = str2double(numClusters);
    [data,numExamples,numFeatures] = scanIn(dataFile);
    [means, variances, priors] = init(data,numClustersNumeric);
    totalLogProb = realmin;
    
    repeat=true;
    while repeat
        [clusterLogDist,clusterLogDistDenominators] = eStep(data,numExamples,numFeatures,numClustersNumeric,means,variances,priors);
        [means, variances, priors] = mStep(data,numExamples,numFeatures,numClustersNumeric,clusterLogDist);
        %exp(clusterLogDist)
        probAfterIteration = totalLikelihoodOfData(numExamples,clusterLogDistDenominators)
        repeat = (probAfterIteration-totalLogProb) > 0.001;
        totalLogProb = probAfterIteration;
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
        max = realmin;
        
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
    
    %{
    %print out all the cluster assignments in order--not right, print off
    %they are. in the wine-true they just happen to be in order
    for c=1:numClusters
       for ex=1:numExamples
           if(assignedCluster(ex) == c)
                fprintf(fid,'%d ',c);
                fprintf(fid,repmat('%f ',[1,numFeatures]),data(ex,:) );
                fprintf(fid,'\n');
           end
       end
    end
    %}
    
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
    maxs( 1 : numFeat ) =  realmin;
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
            means(i,j) = range*rand()+mins(j);
            variances(i,j,j) = range / 2;
        end
    end
    
    %variances(1,:,:)%ensure that variances are diagonal
end

function [clusterLogDist,clusterLogDistDenominators] = eStep(data,numExamples,numFeatures,numClusters,means,variances,priors)
    clusterLogDist = zeros(numExamples,numClusters);
    clusterLogDistDenominators = (1:numExamples).*0;
    logPMax = realmin;
    
    %{
    data
    means
    variances(1,:,:)
    priors
    %}
    
    for ex=1:numExamples
        for c=1:numClusters
            %numerator is ln ( P(Ci) * P(Xk | Ci) )
            %P(Xk | Ci) = multivariate normal distribution
            %http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Density_function
            %apply log to that function, the coeffecient is now added and
            %the exponent of e is now subtracted
            tempVariance =  reshape ( variances(c,:,:),[numFeatures numFeatures] ) ;
            normalLogCoeff = -.5 * log( (2*pi)^double(numExamples) * norm( tempVariance ) );%2pi^ex or nuExamples?
            %as my vectors are row vectors instead of column vectors, the
            %transpose has switched
            %NOTE:inverse tempVariance completed via the divide
            normalLogOfExp = -.5 * ( data(ex,:) - means(c,:) ) / tempVariance *  transpose( data(ex,:)-means(c,:) ) ; 
            
            exp(normalLogCoeff + normalLogOfExp);
            
            %P(Ci) is just from the prior. Add the ln values to get
            %numerator
            clusterLogDist(ex,c) = log(priors(c)) + normalLogCoeff + normalLogOfExp;
            
            %must find the 1 largest numerator to use logsum in the
            %denominator
            if(clusterLogDist(ex,c)>logPMax)
                logPMax=clusterLogDist(ex,c);
            end
        end
    end
    
    %after finding all the numerators & logPMax, log sum must be used to find denomator. 
    for ex=1:numExamples
        %LOGSUM :: log(SUM e^Xi) = Xmax+log(SUM e^(Xi-Xmax))
        logSum=0;
        for c=1:numClusters
            logSum = logSum + exp(clusterLogDist(ex,c)-logPMax);
        end
        
        clusterLogDistDenominators(ex) = logPMax + log(logSum);
        
        %Once denominator has been found, subtract it from every numerator
        for c=1:numClusters
            clusterLogDist(ex,c) = clusterLogDist(ex,c) - clusterLogDistDenominators(ex);
        end
    end
    
    clusterLogDist
    
    %now we have the cluster distributions. The distributions indicate the
    %weight each data point has towards each cluster
end

function [means, variances, priors] = mStep(data,numExamples,numFeatures,numClusters,clusterLogDist)
    %NOTE :safe to convert from clusterLogDist back to probability as the value
    %should be large
    
    %for each cluster prior, average the distribution over every examples
    priors = (1:numClusters).*0;
    
    for c=1:numClusters
        for ex=1:numExamples
            priors(c) = priors(c) + exp(clusterLogDist(ex,c));
        end
        priors(c) = priors(c) / numClusters;
    end
    
    %for each mean, find weighted average of the data
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
    variances = zeros(numClusters,numFeatures,numFeatures);
    for c=1:numClusters
        
        numerators = (1:numFeatures).*0;
        denom=0;
        for ex=1:numExamples
            %for every data feature, find sqaure distance from mean and
            %multiply by probability weight
            weightProb = exp(clusterLogDist(ex,c));
            squareDistFromMean = ( data(ex,:) - means(c,:) ).^2;
            numerators = squareDistFromMean.*weightProb;
            %sum the weights to find the denominator
            denom = denom + weightProb;
        end
        numerators = numerators ./ denom;
        
        %assign the variances to the diagonal of the variance cluster matrix
        for f=1:numFeatures
            variances(c,f,f) = numerators(f);
        end
    end
end

function totalLogProb = totalLikelihoodOfData(numExamples,clusterLogDistDenominators)
    totalLogProb=0;
    
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
    
    %sum up the denominators to find the total    
    for ex=1:numExamples
        totalLogProb = totalLogProb +clusterLogDistDenominators(ex);
    end
    
    %totalLogProb = exp(totalProb)
end
























