
segmentation <- function(dataPath){
  
  datasetName = basename(dataPath)
  csvPath = paste0(dataPath,"/", datasetName,".csv")
  
  # segmentationMethod = "addSegmentOrAugmentPolyDegree"
  # cmdSegmentation = paste0("scala -J-Xmx16g /usr/bin/symbolic ",csvPath," segmentation ", dataPath," ",segmentationMethod)
  # print(cmdSegmentation)
  # system(cmdSegmentation)

  cmdSegmentation = paste("scala -J-Xmx16g ./../../../target/SDLHC-1.0-jar-with-dependencies.jar", csvPath, dataPath)
  print(paste0("running command: ",cmdSegmentation))
  system(cmdSegmentation)
}

removeRedundancy = function(data){
  res = unlist(lapply(data,function(x){gsub('([[:alpha:]])\\1+', '\\1', x)}))
  return(res)
}

dictionaryConstructionAndRecoding <- function(dataPath, dictSize){
  
  coefs = read.csv(paste0(dataPath,"/segmentation/coefficients.csv"))
  limits = read.csv(paste0(dataPath,"/segmentation/segmentLimits.csv"),stringsAsFactors = F)
  
  mod1 <- Mclust(coefs[,3:5],G = dictSize)
  segmentClusters = letters[mod1$classification]
  data = aggregate(segmentClusters, list(coefs$coefId), paste, collapse="")$x
  dataString = data.frame(removeRedundancy(data),stringsAsFactors = F)
  
  return(dataString)
}

categoricalSequenceClusteringAndARI <- function(categoricalSequences, observedClass){
  dist = as.dist(adist(categoricalSequences[,1]))
  nCluster = length(unique(observedClass))
  hclust <- hclust(dist, method = "ward.D2")
  ARI = adj.rand.index(observedClass,cutree(hclust,nCluster))
  return(ARI)
}

InstallAndLoadLibraries <- function(){
  options(warn=-1)
  
  list.of.packages <- c("stringdist","mclust","fossil","dtw","dtwclust")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  lapply(list.of.packages,function(package){eval(parse(text=paste("library(",package,")")))})
  options(warn=0)
  
}


