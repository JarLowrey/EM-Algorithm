function variousClusters(maxNumClusters,dataFile)
    if isa(maxNumClusters,'char')
        maxNumClusters = str2double(maxNumClusters);
    end
    
    for c = 1:maxNumClusters
        logProbsForDifferentClusters(c)  = gaussmix(c,dataFile,'trashFile.txt');
    end
    
    plotClusterLogProbs(logProbsForDifferentClusters,dataFile);
end

function plotClusterLogProbs(logProbsForDifferentClusters,dataFile)

    figure(1);
    sizeOfVector = size(logProbsForDifferentClusters);
    plot((1:sizeOfVector(2)),logProbsForDifferentClusters(1,:) );
    title(dataFile);
    xlabel('Cluster');
    ylabel('Total Log Probability');

end